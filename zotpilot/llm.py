from pathlib import Path
from typing import Any, Generator

from openai import OpenAI

from .embeddings import EmbeddingModel
from .ingestion import process_document
from .retrieval import format_context, similarity_search
from .utils.config import get_openai_api_key, setup_openai_api_key
from .utils.settings import DEFAULT_MAX_TOKENS, DEFAULT_MODEL, DEFAULT_TEMPERATURE


def get_openai_client(api_key: str | None = None) -> OpenAI:
    """
    Get an OpenAI client with appropriate API key.

    Args:
        api_key: Optional API key. If not provided, will use configured key or prompt for one.

    Returns:
        OpenAI client instance
    """
    if not api_key:
        api_key = get_openai_api_key()
        if not api_key:
            api_key = setup_openai_api_key()

    return OpenAI(api_key=api_key)


def build_prompt(query: str, context: str) -> list[dict[str, str]]:
    """
    Build a prompt for the LLM from the query and context.

    Args:
        query: User's question
        context: Context from retrieved document chunks

    Returns:
        list of message dictionaries for OpenAI API
    """
    system_message = (
        "You are a helpful academic research assistant helping with a scientific paper. "
        "Act as an expert in the field of the paper. "
        "Answer questions based on your knowledge of the paper and the context provided. "
        "When citing information, refer to the specific section of the paper where it appears. "
        "Be precise, scholarly, and focus on the factual content of the paper."
    )

    messages = [
        {"role": "system", "content": system_message},
        {
            "role": "user",
            "content": f"Here is the relevant context from the paper:\n\n{context}\n\nQuestion: {query}",
        },
    ]

    return messages


def generate_response(
    query: str,
    context: str,
    model: str = DEFAULT_MODEL,
    temperature: float = DEFAULT_TEMPERATURE,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    client: OpenAI | None = None,
) -> str:
    """
    Generate a response from the LLM for a query with given context.

    Args:
        query: User's question
        context: Context from retrieved document chunks
        model: OpenAI model to use
        temperature: Temperature parameter (0-1)
        max_tokens: Maximum tokens in response
        client: Optional OpenAI client instance

    Returns:
        Generated response from the LLM
    """
    if client is None:
        client = get_openai_client()

    messages = build_prompt(query, context)
    try:
        completion = client.chat.completions.create(
            model=model, messages=messages, temperature=temperature, max_tokens=max_tokens
        )
        return completion.choices[0].message.content
    except Exception as e:
        return f"Error generating response: {e!s}"


def generate_streaming_response(
    query: str,
    context: str,
    model: str = DEFAULT_MODEL,
    temperature: float = DEFAULT_TEMPERATURE,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    client: OpenAI | None = None,
) -> Generator[str, None, None]:
    """
    Generate a streaming response from the LLM.

    Args:
        query: User's question
        context: Context from retrieved document chunks
        model: OpenAI model to use
        temperature: Temperature parameter (0-1)
        max_tokens: Maximum tokens in response
        client: Optional OpenAI client instance

    Yields:
        Chunks of the generated response
    """
    if client is None:
        client = get_openai_client()

    messages = build_prompt(query, context)
    try:
        stream = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
        )

        for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
    except Exception as e:
        yield f"Error generating response: {e!s}"


def process_query(
    query: str,
    retrieved_results: list[dict[str, Any]],
    model: str = DEFAULT_MODEL,
    temperature: float = DEFAULT_TEMPERATURE,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    stream: bool = False,
    client: OpenAI | None = None,
) -> str | Generator[str, None, None]:
    """
    Process a query using the RAG pipeline.

    Args:
        query: User's question
        retrieved_results: Results from the retrieval system
        model: OpenAI model to use
        temperature: Temperature parameter (0-1)
        max_tokens: Maximum tokens in response
        stream: Whether to stream the response
        client: Optional OpenAI client instance

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
        )
    else:
        return generate_response(
            query=query,
            context=context,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            client=client,
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
    client: OpenAI | None = None,
    embedding_model_id: str | None = None,
    api_key: str | None = None,
) -> tuple[str | Generator[str, None, None], list[dict[str, Any]]]:
    """
    Complete RAG pipeline that handles document processing, retrieval, and generation.

    You must provide either document_data (from process_document) or pdf_path.

    Args:
        query: User's question
        document_data: Optional pre-processed document data from process_document
        pdf_path: Path to PDF file (used if document_data is not provided)
        top_k: Number of chunks to retrieve
        model: OpenAI model to use for generation
        temperature: Temperature parameter (0-1)
        max_tokens: Maximum tokens in response
        stream: Whether to stream the response
        embedding_model: Optional pre-initialized embedding model (avoids recreation)
        client: Optional pre-initialized OpenAI client (avoids recreation)
        embedding_model_id: Model ID for embeddings (ignored if embedding_model provided)
        api_key: Optional API key for OpenAI (ignored if client provided)

    Returns:
        Tuple of (generated response or streaming generator, retrieved results)
    """
    if client is None:
        client = get_openai_client(api_key)

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
    )

    return response, retrieved_results
