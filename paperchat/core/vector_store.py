"""Vector storage implementation using ChromaDB.

This module provides a unified interface for storing and retrieving
document chunks and their embeddings using ChromaDB as the backend.
"""

import os
from pathlib import Path
from typing import Any

import chromadb
import numpy as np
import torch
from chromadb.config import Settings

DEFAULT_SIMILARITY_TOP_K = 5
DEFAULT_SIMILARITY_THRESHOLD = 0.7


class VectorStore:
    """Vector storage implementation using ChromaDB.

    This class provides a unified interface for storing document embeddings,
    retrieving similar documents, and managing collections within ChromaDB.
    """

    def __init__(self, persist_directory: str | None = None, create_directory: bool = True):
        """Initialize the VectorStore with a persistence directory.

        Args:
            persist_directory: Directory where ChromaDB will store its data.
                If None, uses ~/.paperchat/vectordb.
            create_directory: Whether to create the persistence directory if it
                doesn't exist.
        """
        self.persist_directory = persist_directory or str(Path.home() / ".paperchat" / "vectordb")

        if create_directory:
            os.makedirs(self.persist_directory, exist_ok=True)

        self.client = chromadb.PersistentClient(
            path=self.persist_directory,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True,
            ),
        )

    def list_collections(self) -> list[str]:
        """List all available collections in the database.

        Returns:
            List of collection names
        """
        collections = self.client.list_collections()
        return [collection.name for collection in collections]

    def create_collection(self, collection_name: str) -> chromadb.Collection:
        """Create a new collection or get existing one with the given name.

        Args:
            collection_name: Name of the collection to create

        Returns:
            ChromaDB collection object
        """
        return self.client.get_or_create_collection(name=collection_name)

    def delete_collection(self, collection_name: str) -> None:
        """Delete a collection from the database.

        Args:
            collection_name: Name of the collection to delete
        """
        self.client.delete_collection(name=collection_name)

    def add_document(
        self,
        collection_name: str,
        chunk_texts: list[str],
        chunk_embeddings: torch.Tensor | np.ndarray,
        chunk_metadata: list[dict[str, Any]],
        ids: list[str] | None = None,
    ) -> None:
        """Add document chunks to a collection.

        Args:
            collection_name: Name of the collection to add to
            chunk_texts: List of text chunks
            chunk_embeddings: Tensor or array of chunk embeddings
            chunk_metadata: List of metadata dictionaries for each chunk
            ids: Optional list of IDs for each chunk. If None, generated automatically.
        """
        collection = self.create_collection(collection_name)

        if isinstance(chunk_embeddings, torch.Tensor):
            embeddings = chunk_embeddings.cpu().numpy()
        else:
            embeddings = chunk_embeddings

        if ids is None:
            ids = [f"{collection_name}_{i}" for i in range(len(chunk_texts))]

        collection.add(
            documents=chunk_texts,
            embeddings=embeddings,
            metadatas=chunk_metadata,
            ids=ids,
        )

    def update_embeddings(
        self,
        collection_name: str,
        ids: list[str],
        embeddings: torch.Tensor | np.ndarray,
    ) -> None:
        """Update embeddings for existing documents in a collection.

        This allows changing the embedding model without re-chunking documents.

        Args:
            collection_name: Name of the collection to update
            ids: List of document IDs to update
            embeddings: New embeddings to assign to the documents
        """
        collection = self.client.get_collection(name=collection_name)

        if isinstance(embeddings, torch.Tensor):
            embeddings_array = embeddings.cpu().numpy()
        else:
            embeddings_array = embeddings

        collection.update(
            ids=ids,
            embeddings=embeddings_array,
        )

    def reembed_collection(
        self,
        collection_name: str,
        embedding_model: Any,
        batch_size: int = 32,
    ) -> None:
        """Re-embed all documents in a collection with a new embedding model.

        This retrieves all documents from the collection, generates new embeddings,
        and updates the collection with the new embeddings.

        Args:
            collection_name: Name of the collection to re-embed
            embedding_model: Model to use for generating new embeddings
                (must have an embed_text method)
            batch_size: Batch size for embedding generation
        """
        collection = self.client.get_collection(name=collection_name)

        result = collection.get()

        if not result["ids"]:
            return

        for i in range(0, len(result["ids"]), batch_size):
            batch_ids = result["ids"][i : i + batch_size]
            batch_texts = result["documents"][i : i + batch_size]

            new_embeddings = embedding_model.embed_text(batch_texts)

            self.update_embeddings(
                collection_name=collection_name,
                ids=batch_ids,
                embeddings=new_embeddings,
            )

    def get_document_texts(self, collection_name: str) -> dict[str, Any]:
        """Get all documents and their metadata from a collection.

        This is useful for retrieving the original texts to re-embed them
        with a different model.

        Args:
            collection_name: Name of the collection

        Returns:
            Dictionary with document IDs, texts, and metadata
        """
        collection = self.client.get_collection(name=collection_name)
        result = collection.get()

        return {
            "ids": result["ids"],
            "documents": result["documents"],
            "metadatas": result["metadatas"],
        }

    def search(
        self,
        query_embedding: torch.Tensor | np.ndarray,
        collection_names: str | list[str] | None = None,
        top_k: int = DEFAULT_SIMILARITY_TOP_K,
        threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
        filter_criteria: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Search for similar chunks across collections.

        Args:
            query_embedding: Embedding of the query
            collection_names: Name or list of collection names to search in.
                If None, searches all collections.
            top_k: Number of results to return per collection
            threshold: Similarity threshold (0-1)
            filter_criteria: Optional filter criteria for metadata

        Returns:
            Dictionary with search results
        """
        if isinstance(query_embedding, torch.Tensor):
            query_embedding = query_embedding.cpu().numpy()

        if isinstance(collection_names, str):
            collection_names = [collection_names]

        if collection_names is None:
            collection_names = self.list_collections()

        results = {}

        for name in collection_names:
            try:
                collection = self.client.get_collection(name=name)

                query_results = collection.query(
                    query_embeddings=query_embedding.reshape(1, -1),
                    n_results=top_k,
                    where=filter_criteria,
                )

                # Filter by threshold
                filtered_indices = []
                for i, distance in enumerate(query_results["distances"][0]):
                    similarity = 1.0 - distance
                    if similarity >= threshold:
                        filtered_indices.append(i)

                results[name] = {
                    "documents": [query_results["documents"][0][i] for i in filtered_indices],
                    "metadatas": [query_results["metadatas"][0][i] for i in filtered_indices],
                    "distances": [query_results["distances"][0][i] for i in filtered_indices],
                    "ids": [query_results["ids"][0][i] for i in filtered_indices],
                }

            except ValueError:
                # Collection doesn't exist, skip
                continue

        return results

    def get_collection_info(self, collection_name: str) -> dict[str, Any]:
        """Get information about a collection.

        Args:
            collection_name: Name of the collection

        Returns:
            Dictionary with collection information
        """
        collection = self.client.get_collection(name=collection_name)
        return {
            "name": collection_name,
            "count": collection.count(),
        }
