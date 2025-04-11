from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Generator, List

from .embeddings import EmbeddingModel
from .ingestion import process_document
from .llms.manager import LLMManager
from .retrieval import similarity_search
from .utils.config import DEFAULT_MAX_TOKENS, DEFAULT_MODEL, DEFAULT_PROVIDER, DEFAULT_TEMPERATURE
from .utils.formatting import format_context


@dataclass
class LLMConfig:
    """Configuration class for LLM settings."""

    provider_name: str = DEFAULT_PROVIDER
    model_id: str = DEFAULT_MODEL
    temperature: float = DEFAULT_TEMPERATURE
    max_tokens: int = DEFAULT_MAX_TOKENS

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "LLMConfig":
        """Create configuration from dictionary."""
        return cls(**{k: v for k, v in config_dict.items() if k in cls.__dataclass_fields__})


def rag_pipeline(
    query: str,
    document_data: Dict[str, Any] | None = None,
    pdf_path: str | Path | None = None,
    top_k: int = 5,
    stream: bool = False,
    embedding_model: Any | None = None,
    embedding_model_id: str | None = None,
    llm_manager: LLMManager | None = None,
) -> tuple[str | Generator[str, None, None], List[Dict[str, Any]]]:
    """
    Complete RAG pipeline that handles document processing, retrieval, and generation.

    You must provide either document_data (from process_document) or pdf_path.

    Args:
        query: User's question
        document_data: Optional pre-processed document data from process_document
        pdf_path: Path to PDF file (used if document_data is not provided)
        top_k: Number of chunks to retrieve
        stream: Whether to stream the response
        embedding_model: Optional pre-initialized embedding model (avoids recreation)
        embedding_model_id: Model ID for embeddings (ignored if embedding_model provided)
        llm_manager: Optional LLMManager instance (avoids recreation)

    Returns:
        Tuple of (generated response or streaming generator, retrieved results)
    """
    if embedding_model is None:
        embedding_model = EmbeddingModel(model_id=embedding_model_id)

    if document_data is None:
        if pdf_path is None:
            raise ValueError("Either document_data or pdf_path must be provided")

        document_data = process_document(pdf_path, embedding_model=embedding_model)

    # Perform vector similarity search
    query_embedding = embedding_model.embed_text([query])[0]
    retrieved_results = similarity_search(
        query_embedding=query_embedding,
        chunk_texts=document_data["chunk_texts"],
        chunk_embeddings=document_data["chunk_embeddings"],
        chunk_metadata=document_data["chunk_metadata"],
        top_k=top_k,
    )

    # Generate LLM response
    context = format_context(retrieved_results, include_metadata=True)
    if llm_manager is None:
        llm_manager = LLMManager()

    if stream:
        response = llm_manager.generate_streaming_response(query, context)
    else:
        response = llm_manager.generate_response(query, context)

    return response, retrieved_results
