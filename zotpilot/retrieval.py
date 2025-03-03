from typing import Any

import torch

from .embeddings import get_text_embeddings


def similarity_search(
    query_embedding: torch.Tensor,
    chunk_embeddings: torch.Tensor,
    chunk_texts: list[str],
    chunk_metadata: list[dict[str, Any]],
    top_k: int = 5,
) -> list[dict[str, Any]]:
    """
    Perform in-memory similarity search using cosine similarity.

    Args:
        query_embedding: Embedding of the query
        chunk_embeddings: Embeddings of all chunks
        chunk_texts: Texts of all chunks
        chunk_metadata: Metadata for all chunks
        top_k: Number of results to return

    Returns:
        List of dicts containing matched chunks with their metadata and similarity scores
    """
    similarities = torch.nn.functional.cosine_similarity(
        query_embedding.unsqueeze(0), chunk_embeddings
    )
    top_k = min(top_k, len(chunk_texts))
    top_indices = similarities.argsort(descending=True)[:top_k].tolist()
    results = []
    for i in top_indices:
        results.append(
            {
                "text": chunk_texts[i],
                "metadata": chunk_metadata[i],
                "similarity": similarities[i].item(),
                "distance": 1.0 - similarities[i].item(),
            }
        )

    return results


def search_chunks(
    query: str,
    chunk_texts: list[str],
    chunk_embeddings: torch.Tensor,
    chunk_metadata: list[dict[str, Any]],
    top_k: int = 5,
    model_id: str | None = None,
) -> list[dict[str, Any]]:
    """
    Search for chunks relevant to a query.

    Args:
        query: Query text
        chunk_texts: List of chunk texts
        chunk_embeddings: Tensor of chunk embeddings
        chunk_metadata: List of chunk metadata
        top_k: Number of results to return
        model_id: Model ID for embeddings

    Returns:
        List of dicts containing matched chunks with their metadata and similarity scores
    """
    query_embedding = get_text_embeddings([query], model_id=model_id)[0]
    results = similarity_search(
        query_embedding=query_embedding,
        chunk_embeddings=chunk_embeddings,
        chunk_texts=chunk_texts,
        chunk_metadata=chunk_metadata,
        top_k=top_k,
    )

    return results


def format_context(results: list[dict[str, Any]], include_metadata: bool = True) -> str:
    """
    Format retrieved chunks into a context string for the LLM.

    Args:
        results: List of retrieval results
        include_metadata: Whether to include metadata in the context

    Returns:
        Formatted context string
    """
    context_parts = []

    for i, result in enumerate(results):
        if include_metadata:
            context_parts.append(
                f"[CHUNK {i + 1} (Page {result['metadata']['page']})]:\n{result['text']}\n"
            )
        else:
            context_parts.append(result["text"])

    return "\n\n".join(context_parts)
