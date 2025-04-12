from pathlib import Path
from typing import Any, Generator

from ..llms.manager import LLMManager
from ..utils.formatting import format_context
from .embeddings import EmbeddingModel
from .ingestion import process_document
from .retrieval import similarity_search


class RAGPipeline:
    """
    Orchestrates the Retrieval-Augmented Generation (RAG) pipeline.

    Handles document processing, retrieval of relevant chunks, and generation
    of responses using a configured LLM.
    """

    def __init__(self, llm_manager: LLMManager, embedding_model: EmbeddingModel):
        """
        Initializes the RAG pipeline with necessary components.

        Args:
            llm_manager: An instance of LLMManager to handle LLM interactions.
            embedding_model: An instance of EmbeddingModel for text embeddings.
        """
        self.llm_manager = llm_manager
        self.embedding_model = embedding_model

    def _ensure_document_data(
        self, document_data: dict[str, Any] | None = None, pdf_path: str | Path | None = None
    ) -> dict[str, Any]:
        """
        Ensures document data is available, processing PDF if necessary.

        Raises:
            ValueError: If neither document_data nor pdf_path is provided.
        """
        if document_data:
            return document_data
        if pdf_path:
            return process_document(pdf_path, embedding_model=self.embedding_model)

        raise ValueError("Either document_data or pdf_path must be provided")

    def retrieve(
        self, query: str, document_data: dict[str, Any], top_k: int = 5
    ) -> list[dict[str, Any]]:
        """
        Retrieves relevant document chunks based on the query.

        Args:
            query: The user's query text.
            document_data: Processed document data containing texts and embeddings.
            top_k: The number of chunks to retrieve.

        Returns:
            A list of retrieved document chunk dictionaries with metadata.
        """
        query_embedding = self.embedding_model.embed_text([query])[0]

        return similarity_search(
            query_embedding=query_embedding,
            chunk_texts=document_data["chunk_texts"],
            chunk_embeddings=document_data["chunk_embeddings"],
            chunk_metadata=document_data["chunk_metadata"],
            top_k=top_k,
        )

    def generate(
        self, query: str, retrieved_results: list[dict[str, Any]], stream: bool = False
    ) -> str | Generator[str, None, None]:
        """
        Generates a response using the LLM based on the query and retrieved context.

        Args:
            query: The user's query text.
            retrieved_results: List of document chunks retrieved as context.
            stream: Whether to stream the response.

        Returns:
            The generated response (str or Generator).
        """
        context = format_context(retrieved_results, include_metadata=True)

        if stream:
            return self.llm_manager.generate_streaming_response(query, context)
        else:
            return self.llm_manager.generate_response(query, context)

    def run(
        self,
        query: str,
        document_data: dict[str, Any] | None = None,
        pdf_path: str | Path | None = None,
        top_k: int = 5,
        stream: bool = False,
    ) -> tuple[str | Generator[str, None, None], list[dict[str, Any]]]:
        """
        Executes the full RAG pipeline: process/ensure doc data, retrieve, generate.

        Args:
            query: The user's query.
            document_data: Optional pre-processed document data.
            pdf_path: Optional path to a PDF file (used if document_data is None).
            top_k: Number of chunks to retrieve.
            stream: Whether to stream the LLM response.

        Returns:
            A tuple containing the generated response and the list of retrieved chunks.
        """
        processed_data = self._ensure_document_data(document_data, pdf_path)
        retrieved_chunks = self.retrieve(query, processed_data, top_k)
        response = self.generate(query, retrieved_chunks, stream)

        return response, retrieved_chunks
