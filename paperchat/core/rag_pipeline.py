from typing import Any, Generator

from ..llms.manager import LLMManager
from ..utils.formatting import format_context
from .vector_store import DEFAULT_SIMILARITY_THRESHOLD, VectorStore


class RAGPipeline:
    """
    Orchestrates the Retrieval-Augmented Generation (RAG) pipeline.

    Handles document processing, retrieval of relevant chunks, and generation
    of responses using a configured LLM.
    """

    def __init__(self, llm_manager: LLMManager):
        """
        Initializes the RAG pipeline with necessary components.

        Args:
            llm_manager: An instance of LLMManager to handle LLM interactions.
        """
        self.llm_manager = llm_manager
        self.vector_store = VectorStore()

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
        filter_expression: str | None = None,
    ) -> list[dict[str, Any]]:
        """
        Retrieves relevant document chunks based on the query.

        Args:
            query: The user's query text.
            top_k: The number of chunks to retrieve.
            threshold: The minimum similarity score to return.
            filter_expression: A filter expression to apply to the query.
        Returns:
            A list of retrieved document chunk dictionaries with metadata.
        """
        # returns a Milvus ExtraList where the actual list[dict] is the first element
        results = self.vector_store.retrieve(
            query_text=query,
            top_k=top_k,
            threshold=threshold,
            filter_expression=filter_expression,
        )
        return results[0]

    def generate(
        self, query: str, retrieved_results: list[dict[str, Any]], stream: bool = False
    ) -> str | Generator[str, None, None]:
        """
        Generates a response using the LLM based on the query and retrieved context.

        Args:
            query: The user's query text.
            retrieved_results: List of document text and metadata retrieved as context.
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
        retrieved_chunks = self.retrieve(query, top_k)
        response = self.generate(query, retrieved_chunks, stream)

        return response, retrieved_chunks
