"""Vector storage implementation using ChromaDB.

This module provides a unified interface for storing and retrieving
document chunks and their embeddings using ChromaDB as the backend.
"""

import datetime
import os
from pathlib import Path
from typing import Any

import chromadb
import numpy as np
import torch
from chromadb.config import Settings

from paperchat.utils.logging import get_component_logger

logger = get_component_logger("vector_store")

DEFAULT_SIMILARITY_TOP_K = 5
DEFAULT_SIMILARITY_THRESHOLD = 0.7
VECTOR_DB_DIR = str(Path.home() / ".paperchat" / "vectordb")
DOCUMENT_REGISTRY = "document_registry"


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
        self.persist_directory = persist_directory or VECTOR_DB_DIR
        logger.info(f"Initializing VectorStore with directory: {self.persist_directory}")

        if create_directory:
            os.makedirs(self.persist_directory, exist_ok=True)
            logger.debug(f"Created vector store directory: {self.persist_directory}")

        self.client = chromadb.PersistentClient(
            path=self.persist_directory,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True,
            ),
        )
        logger.debug(f"ChromaDB client initialized with path: {self.persist_directory}")

    def list_collections(self) -> list[str]:
        """List all available collections in the database.

        Returns:
            List of collection names
        """
        collections = self.client.list_collections()
        collection_names = [collection.name for collection in collections]
        logger.debug(f"Found collections: {collection_names}")
        return collection_names

    def create_collection(self, collection_name: str) -> chromadb.Collection:
        """Create a new collection or get existing one with the given name.

        Args:
            collection_name: Name of the collection to create

        Returns:
            ChromaDB collection object
        """
        logger.debug(f"Creating/getting collection: {collection_name}")
        return self.client.get_or_create_collection(name=collection_name)

    def delete_collection(self, collection_name: str) -> None:
        """Delete a collection from the database.

        Args:
            collection_name: Name of the collection to delete
        """
        logger.info(f"Deleting collection: {collection_name}")
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
        logger.info(f"Adding document to collection '{collection_name}': {len(chunk_texts)} chunks")
        collection = self.create_collection(collection_name)

        if isinstance(chunk_embeddings, torch.Tensor):
            embeddings = chunk_embeddings.cpu().numpy()
        else:
            embeddings = chunk_embeddings

        if ids is None:
            ids = [f"{collection_name}_{i}" for i in range(len(chunk_texts))]

        # Log metadata structure at debug level
        if chunk_metadata and chunk_metadata[0]:
            logger.debug(f"Metadata structure (first item): {list(chunk_metadata[0].keys())}")

        # Verify we don't have list values in metadata which ChromaDB doesn't support
        for i, metadata in enumerate(chunk_metadata):
            for key, value in list(metadata.items()):
                if isinstance(value, list):
                    logger.debug(f"Converting list to string in metadata[{i}][{key}]")
                    chunk_metadata[i][key] = ", ".join(map(str, value)) if value else ""

        collection.add(
            documents=chunk_texts,
            embeddings=embeddings,
            metadatas=chunk_metadata,
            ids=ids,
        )
        logger.info(f"Successfully added document to collection '{collection_name}'")

    def register_document(
        self,
        document_hash: str,
        original_filename: str,
        collection_name: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Register a document in the document registry.

        This stores metadata about documents without duplicating the files.

        Args:
            document_hash: Hash of the document content
            original_filename: Original filename
            collection_name: Name of the collection where chunks are stored
            metadata: Additional metadata to store with the document
        """
        # Create a document registry collection if it doesn't exist
        registry = self.create_collection(DOCUMENT_REGISTRY)

        # Create metadata for the document
        doc_metadata = {
            "original_filename": original_filename,
            "collection_name": collection_name,
            "registered_at": datetime.datetime.now().isoformat(),
        }

        # Add any additional metadata
        if metadata:
            doc_metadata.update(metadata)

        # Store in the registry - using document_hash as the ID
        registry.upsert(
            ids=[document_hash],
            documents=[f"Document: {original_filename}"],
            metadatas=[doc_metadata],
        )
        logger.info(f"Registered document: {original_filename} (hash: {document_hash})")

    def get_document_info(self, document_hash: str) -> tuple[bool, dict[str, Any]]:
        """Get information about a registered document.

        Args:
            document_hash: Hash of the document content

        Returns:
            Tuple of (exists, metadata)
        """
        try:
            if DOCUMENT_REGISTRY not in self.list_collections():
                return False, {}

            registry = self.client.get_collection(DOCUMENT_REGISTRY)
            result = registry.get(ids=[document_hash])

            if result and result["ids"]:
                return True, result["metadatas"][0]
            return False, {}
        except Exception as e:
            logger.warning(f"Error getting document info: {e}")
            return False, {}

    def list_registered_documents(self) -> list[dict[str, Any]]:
        """List all registered documents.

        Returns:
            List of document metadata dictionaries
        """
        try:
            if DOCUMENT_REGISTRY not in self.list_collections():
                return []

            registry = self.client.get_collection(DOCUMENT_REGISTRY)
            result = registry.get()

            documents = []
            for i, doc_id in enumerate(result["ids"]):
                doc_info = {"document_hash": doc_id, **result["metadatas"][i]}
                documents.append(doc_info)

            return documents
        except Exception as e:
            logger.warning(f"Error listing documents: {e}")
            return []

    def unregister_document(self, document_hash: str) -> bool:
        """Remove a document from the registry.

        This doesn't remove the actual collection with chunks.

        Args:
            document_hash: Hash of the document to unregister

        Returns:
            True if document was found and unregistered, False otherwise
        """
        try:
            if DOCUMENT_REGISTRY not in self.list_collections():
                return False

            exists, metadata = self.get_document_info(document_hash)
            if not exists:
                return False

            registry = self.client.get_collection(DOCUMENT_REGISTRY)
            registry.delete(ids=[document_hash])
            logger.info(
                f"Unregistered document: {metadata.get('original_filename', document_hash)}"
            )
            return True
        except Exception as e:
            logger.warning(f"Error unregistering document: {e}")
            return False

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
        logger.info(
            f"Updating embeddings for {len(ids)} documents in collection '{collection_name}'"
        )
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
        logger.info(f"Re-embedding collection '{collection_name}'")
        collection = self.client.get_collection(name=collection_name)

        result = collection.get()

        if not result["ids"]:
            logger.warning(f"Collection '{collection_name}' is empty, nothing to re-embed")
            return

        logger.info(f"Found {len(result['ids'])} documents to re-embed")

        for i in range(0, len(result["ids"]), batch_size):
            batch_ids = result["ids"][i : i + batch_size]
            batch_texts = result["documents"][i : i + batch_size]

            logger.debug(
                f"Re-embedding batch {i // batch_size + 1}/{(len(result['ids']) - 1) // batch_size + 1}"
            )
            new_embeddings = embedding_model.embed_text(batch_texts)

            self.update_embeddings(
                collection_name=collection_name,
                ids=batch_ids,
                embeddings=new_embeddings,
            )

        logger.info(f"Re-embedding of collection '{collection_name}' complete")

    def get_document_texts(self, collection_name: str) -> dict[str, Any]:
        """Get all documents and their metadata from a collection.

        This is useful for retrieving the original texts to re-embed them
        with a different model.

        Args:
            collection_name: Name of the collection

        Returns:
            Dictionary with document IDs, texts, and metadata
        """
        logger.info(f"Getting document texts from collection '{collection_name}'")
        collection = self.client.get_collection(name=collection_name)
        result = collection.get()
        logger.info(f"Retrieved {len(result['ids'])} documents from collection '{collection_name}'")

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
        logger.debug(f"Searching collections with top_k={top_k}, threshold={threshold}")
        if isinstance(query_embedding, torch.Tensor):
            query_embedding = query_embedding.cpu().numpy()

        if isinstance(collection_names, str):
            collection_names = [collection_names]

        if collection_names is None:
            collection_names = self.list_collections()

        # Filter out the document registry from search collections
        if DOCUMENT_REGISTRY in collection_names:
            collection_names.remove(DOCUMENT_REGISTRY)

        logger.debug(f"Searching in collections: {collection_names}")
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
                logger.debug(f"Found {len(filtered_indices)} results in collection '{name}'")

            except ValueError:
                # Collection doesn't exist, skip
                logger.warning(f"Collection '{name}' not found, skipping")
                continue

        logger.info(f"Search completed across {len(collection_names)} collections")
        return results

    def get_collection_info(self, collection_name: str) -> dict[str, Any]:
        """Get information about a collection.

        Args:
            collection_name: Name of the collection

        Returns:
            Dictionary with collection information
        """
        logger.debug(f"Getting info for collection '{collection_name}'")
        collection = self.client.get_collection(name=collection_name)
        count = collection.count()
        logger.debug(f"Collection '{collection_name}' has {count} items")
        return {
            "name": collection_name,
            "count": count,
        }
