from pathlib import Path
from typing import Any, Generator

from .embeddings import EmbeddingModel
from .ingestion import process_document
from .llms.common import generate_response, generate_streaming_response, get_client
from .retrieval import similarity_search
from .utils.formatting import format_context, format_response_with_citations
from .utils.settings import DEFAULT_MAX_TOKENS, DEFAULT_MODEL, DEFAULT_PROVIDER, DEFAULT_TEMPERATURE


def process_query(
    query: str,
    retrieved_results: list[dict[str, Any]],
    model: str = DEFAULT_MODEL,
    temperature: float = DEFAULT_TEMPERATURE,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    stream: bool = False,
    client: Any | None = None,
    provider_name: str = DEFAULT_PROVIDER,
) -> str | Generator[str, None, None]:
    """
    Process a query using the RAG pipeline.

    Args:
        query: User's question
        retrieved_results: Results from the retrieval system
        model: Model to use
        temperature: Temperature parameter (0-1)
        max_tokens: Maximum tokens in response
        stream: Whether to stream the response
        client: Optional client instance
        provider_name: Name of the provider to use

    Returns:
        Generated response or streaming generator
    """
    context = format_context(retrieved_results, include_metadata=True)
    if stream:
        return generate_streaming_response(
            query=query,
            context=context,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            client=client,
            provider_name=provider_name,
        )
    else:
        return generate_response(
            query=query,
            context=context,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            client=client,
            provider_name=provider_name,
        )


def rag_pipeline(
    query: str,
    document_data: dict[str, Any] | None = None,
    pdf_path: str | Path | None = None,
    top_k: int = 5,
    model: str = DEFAULT_MODEL,
    temperature: float = DEFAULT_TEMPERATURE,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    stream: bool = False,
    embedding_model: Any = None,
    client: Any | None = None,
    embedding_model_id: str | None = None,
    api_key: str | None = None,
    provider_name: str = DEFAULT_PROVIDER,
) -> tuple[str | Generator[str, None, None], list[dict[str, Any]]]:
    """
    Complete RAG pipeline that handles document processing, retrieval, and generation.

    You must provide either document_data (from process_document) or pdf_path.

    Args:
        query: User's question
        document_data: Optional pre-processed document data from process_document
        pdf_path: Path to PDF file (used if document_data is not provided)
        top_k: Number of chunks to retrieve
        model: Model to use for generation
        temperature: Temperature parameter (0-1)
        max_tokens: Maximum tokens in response
        stream: Whether to stream the response
        embedding_model: Optional pre-initialized embedding model (avoids recreation)
        client: Optional pre-initialized client (avoids recreation)
        embedding_model_id: Model ID for embeddings (ignored if embedding_model provided)
        api_key: Optional API key
        provider_name: Name of the provider to use

    Returns:
        Tuple of (generated response or streaming generator, retrieved results)
    """
    if client is None:
        client = get_client(provider_name=provider_name, api_key=api_key)

    if embedding_model is None:
        embedding_model = EmbeddingModel(model_id=embedding_model_id)

    if document_data is None:
        if pdf_path is None:
            raise ValueError("Either document_data or pdf_path must be provided")

        document_data = process_document(pdf_path, embedding_model=embedding_model)

    query_embedding = embedding_model.embed_text([query])[0]
    retrieved_results = similarity_search(
        query_embedding=query_embedding,
        chunk_texts=document_data["chunk_texts"],
        chunk_embeddings=document_data["chunk_embeddings"],
        chunk_metadata=document_data["chunk_metadata"],
        top_k=top_k,
    )
    response = process_query(
        query=query,
        retrieved_results=retrieved_results,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        stream=stream,
        client=client,
        provider_name=provider_name,
    )

    response = format_response_with_citations(response)

    return response, retrieved_results
