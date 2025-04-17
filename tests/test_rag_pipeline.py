"""Tests for the llm.py module that contains the RAGPipeline."""

from unittest import mock

import pytest

from paperchat.core import RAGPipeline


@pytest.fixture
def mock_llm_manager():
    """Create a mock LLMManager instance."""
    return mock.MagicMock()


@pytest.fixture
def mock_vector_store():
    """Create a mock VectorStore instance."""
    with mock.patch("paperchat.core.VectorStore") as mock_vs_cls:
        mock_vs = mock.MagicMock()
        mock_vs_cls.return_value = mock_vs
        yield mock_vs


@pytest.fixture
def rag_pipeline(mock_llm_manager, mock_vector_store):
    """Fixture to create a RAGPipeline instance with mocked dependencies."""
    with mock.patch("paperchat.core.rag_pipeline.VectorStore", return_value=mock_vector_store):
        pipeline = RAGPipeline(llm_manager=mock_llm_manager)
    return pipeline


def test_init(mock_llm_manager):
    """Test initialization of RAGPipeline."""
    with mock.patch("paperchat.core.rag_pipeline.VectorStore") as mock_vs_cls:
        mock_vs_instance = mock.MagicMock()
        mock_vs_cls.return_value = mock_vs_instance

        pipeline = RAGPipeline(llm_manager=mock_llm_manager)

        assert pipeline.llm_manager is mock_llm_manager

        mock_vs_cls.assert_called_once()
        assert pipeline.vector_store is mock_vs_instance


def test_retrieve(rag_pipeline, mock_vector_store):
    """Test the retrieve method."""
    query = "test query"
    top_k = 3
    threshold = 0.8
    filter_expression = "source == 'test.pdf'"

    mock_inner_results = [{"text": "test result"}]
    mock_vector_store.retrieve.return_value = [mock_inner_results]

    results = rag_pipeline.retrieve(query, top_k, threshold, filter_expression)

    assert results == mock_inner_results
    mock_vector_store.retrieve.assert_called_once_with(
        query_text=query,
        top_k=top_k,
        threshold=threshold,
        filter_expression=filter_expression,
    )


def test_generate_non_streaming(rag_pipeline, mock_llm_manager):
    """Test the generate method in non-streaming mode."""
    query = "test query"
    retrieved_results = [
        {"text": "context 1", "source": "doc1"},
        {"text": "context 2", "source": "doc2"},
    ]
    expected_response = "This is the answer."

    with mock.patch("paperchat.core.rag_pipeline.format_context") as mock_format_context:
        mock_format_context.return_value = "formatted context"
        mock_llm_manager.generate_response.return_value = expected_response

        response = rag_pipeline.generate(query, retrieved_results, stream=False)

        assert response == expected_response
        mock_format_context.assert_called_once_with(retrieved_results, include_metadata=True)
        mock_llm_manager.generate_response.assert_called_once_with(query, "formatted context")
        mock_llm_manager.generate_streaming_response.assert_not_called()


def test_generate_streaming(rag_pipeline, mock_llm_manager):
    """Test the generate method in streaming mode."""
    query = "test query"
    retrieved_results = [
        {"text": "context 1", "source": "doc1"},
        {"text": "context 2", "source": "doc2"},
    ]

    mock_generator = mock.MagicMock()

    with mock.patch("paperchat.core.rag_pipeline.format_context") as mock_format_context:
        mock_format_context.return_value = "formatted context"
        mock_llm_manager.generate_streaming_response.return_value = mock_generator

        response = rag_pipeline.generate(query, retrieved_results, stream=True)

        assert response == mock_generator
        mock_format_context.assert_called_once_with(retrieved_results, include_metadata=True)
        mock_llm_manager.generate_streaming_response.assert_called_once_with(
            query, "formatted context"
        )
        mock_llm_manager.generate_response.assert_not_called()


def test_run_non_streaming(rag_pipeline):
    """Test the run method in non-streaming mode."""
    query = "test query"
    top_k = 3

    retrieved_chunks = [{"text": "chunk 1"}]
    expected_response = "This is the answer."

    rag_pipeline.retrieve = mock.MagicMock(return_value=retrieved_chunks)
    rag_pipeline.generate = mock.MagicMock(return_value=expected_response)

    response, chunks = rag_pipeline.run(query, top_k, stream=False)

    assert response == expected_response
    assert chunks == retrieved_chunks
    rag_pipeline.retrieve.assert_called_once_with(query, top_k)
    rag_pipeline.generate.assert_called_once_with(query, retrieved_chunks, False)


def test_run_streaming(rag_pipeline):
    """Test the run method in streaming mode."""
    query = "test query"
    top_k = 3

    retrieved_chunks = [{"text": "chunk 1"}]
    mock_generator = mock.MagicMock()

    rag_pipeline.retrieve = mock.MagicMock(return_value=retrieved_chunks)
    rag_pipeline.generate = mock.MagicMock(return_value=mock_generator)

    response, chunks = rag_pipeline.run(query, top_k, stream=True)

    assert response == mock_generator
    assert chunks == retrieved_chunks
    rag_pipeline.retrieve.assert_called_once_with(query, top_k)
    rag_pipeline.generate.assert_called_once_with(query, retrieved_chunks, True)
