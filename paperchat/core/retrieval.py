from typing import Any

import torch


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
