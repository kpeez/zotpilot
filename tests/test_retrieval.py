import pytest
import torch

from paperchat.core import similarity_search
from paperchat.utils.formatting import format_context


def test_similarity_search():
    """Test the similarity search function with controlled inputs"""
    # test data: unit vector in x direction
    query_embedding = torch.tensor([1.0, 0.0, 0.0])
    # chunk embeddings: unit vectors in x, y, and z directions
    chunk_embeddings = torch.tensor(
        [
            [1.0, 0.0, 0.0],  # perfect match (cosine similarity = 1.0)
            [0.0, 1.0, 0.0],  # orthogonal (cosine similarity = 0.0)
            [0.7, 0.7, 0.0],  # partial match (cosine similarity = 0.7)
        ]
    )
    chunk_texts = ["Document A", "Document B", "Document C"]
    chunk_metadata = [
        {"page": 1},
        {"page": 2},
        {"page": 3},
    ]
    results = similarity_search(
        query_embedding=query_embedding,
        chunk_embeddings=chunk_embeddings,
        chunk_texts=chunk_texts,
        chunk_metadata=chunk_metadata,
        top_k=3,
    )

    assert len(results) == 3
    # check correct order (by similarity)
    assert results[0]["text"] == "Document A"
    assert results[1]["text"] == "Document C"
    assert results[2]["text"] == "Document B"
    # check similarity values
    assert results[0]["similarity"] == pytest.approx(1.0)
    assert results[1]["similarity"] > 0.5  # should be around 0.7
    assert results[2]["similarity"] == pytest.approx(0.0)

    # test with limited top_k
    results_limited = similarity_search(
        query_embedding=query_embedding,
        chunk_embeddings=chunk_embeddings,
        chunk_texts=chunk_texts,
        chunk_metadata=chunk_metadata,
        top_k=1,
    )

    assert len(results_limited) == 1
    assert results_limited[0]["text"] == "Document A"
    # Test with empty inputs
    empty_results = similarity_search(
        query_embedding=query_embedding,
        chunk_embeddings=torch.zeros((0, 3)),
        chunk_texts=[],
        chunk_metadata=[],
    )

    assert len(empty_results) == 0


def test_format_context():
    """Test the format_context function with various inputs"""
    # Create test data
    results = [
        {
            "text": "This is the first chunk.",
            "metadata": {"page": 1},
            "similarity": 0.9,
        },
        {
            "text": "This is the second chunk.",
            "metadata": {"page": 5},
            "similarity": 0.7,
        },
    ]

    formatted_context = format_context(results)

    assert "[CHUNK 1 (Page 1)]:" in formatted_context
    assert "This is the first chunk." in formatted_context
    assert "[CHUNK 2 (Page 5)]:" in formatted_context
    assert "This is the second chunk." in formatted_context

    # test without metadata
    formatted_context_no_metadata = format_context(results, include_metadata=False)

    assert "[CHUNK" not in formatted_context_no_metadata
    assert "(Page" not in formatted_context_no_metadata
    assert "This is the first chunk." in formatted_context_no_metadata
    assert "This is the second chunk." in formatted_context_no_metadata

    # test with empty results
    empty_context = format_context([])
    assert empty_context == ""
    # test with missing page metadata
    results_missing_page = [
        {
            "text": "Missing page metadata.",
            "metadata": {},  # Missing "page" key
            "similarity": 0.8,
        }
    ]

    with pytest.raises(KeyError):
        format_context(results_missing_page)
