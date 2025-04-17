"""Tests for the vector store module."""

import tempfile
from pathlib import Path
from unittest import mock

import numpy as np
import pytest
from pymilvus import CollectionSchema

from paperchat.core.vector_store import VectorStore


@pytest.fixture
def temp_db_path():
    """Create a temporary directory for the test database."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield str(Path(tmpdir) / "paperchat.db")


@pytest.fixture
def mock_milvus_client():
    """Create a mock MilvusClient."""
    with mock.patch("paperchat.core.vector_store.MilvusClient") as mock_client_class:
        mock_client = mock.MagicMock()
        mock_client_class.return_value = mock_client
        yield mock_client


@pytest.fixture
def vector_store(mock_milvus_client):
    """Create a VectorStore instance with a mocked MilvusClient."""
    with mock.patch("paperchat.core.vector_store.get_component_logger") as mock_get_logger:
        mock_get_logger.return_value = mock.MagicMock()
        store = VectorStore()
        store.client = mock_milvus_client
        yield store


def test_init_connects_to_milvus():
    """Test that initialization connects to Milvus."""
    with (
        mock.patch("paperchat.core.vector_store.MilvusClient") as mock_client_class,
        mock.patch("paperchat.core.vector_store.VectorStore.build_index"),
        mock.patch("paperchat.core.vector_store.VectorStore.get_or_create_collection"),
        mock.patch("paperchat.core.vector_store.get_component_logger") as mock_get_logger,
    ):
        mock_get_logger.return_value = mock.MagicMock()
        mock_client = mock.MagicMock()
        mock_client_class.return_value = mock_client

        store = VectorStore()

        expected_db_path = str(Path.home() / ".paperchat" / "paperchat.db")
        assert store.db_path == expected_db_path
        assert store.collection_name == "paperchat_docs"
        mock_client_class.assert_called_once_with(uri=store.db_path)


def test_define_schema(vector_store):
    """Test that the schema is defined correctly."""
    schema = vector_store._define_schema()

    assert isinstance(schema, CollectionSchema)
    field_names = [field.name for field in schema.fields]
    assert "pk" in field_names
    assert "chunk_id" in field_names
    assert "embedding" in field_names
    assert "text" in field_names
    assert "label" in field_names
    assert "page_numbers" in field_names
    assert "headings" in field_names
    assert "source" in field_names


def test_get_or_create_collection(vector_store):
    """Test creating a collection if it doesn't exist."""
    vector_store.client.reset_mock()

    vector_store.client.has_collection.return_value = False

    vector_store.get_or_create_collection()

    vector_store.client.has_collection.assert_called_once_with(vector_store.collection_name)
    vector_store.client.create_collection.assert_called_once_with(
        collection_name=vector_store.collection_name, schema=vector_store.schema
    )


def test_get_or_create_collection_existing(vector_store):
    """Test not creating a collection if it already exists."""
    vector_store.client.reset_mock()
    vector_store.client.has_collection.return_value = True
    vector_store.get_or_create_collection()
    vector_store.client.has_collection.assert_called_once_with(vector_store.collection_name)
    vector_store.client.create_collection.assert_not_called()


def test_build_index(vector_store):
    """Test building an index."""
    vector_store.client.reset_mock()

    mock_index_params = mock.MagicMock()
    vector_store.client.prepare_index_params.return_value = mock_index_params

    vector_store.build_index()

    vector_store.client.prepare_index_params.assert_called_once()
    mock_index_params.add_index.assert_called_once_with(
        field_name="embedding", index_type="IVF_FLAT", metric_type="COSINE", params={"nlist": 128}
    )
    vector_store.client.create_index.assert_called_once_with(
        collection_name=vector_store.collection_name, index_params=mock_index_params
    )


def test_add_document(vector_store):
    """Test adding a document."""
    pdf_path = "/path/to/test.pdf"
    mock_process_result = [{"chunk_id": "test", "embedding": [0.1, 0.2], "text": "test text"}]

    # Mock the check for existing document
    vector_store.client.query.return_value = []

    with mock.patch(
        "paperchat.core.vector_store.process_pdf_document", return_value=mock_process_result
    ):
        result = vector_store.add_document(pdf_path)

        assert result is True
        vector_store.client.query.assert_called_once()
        vector_store.client.insert.assert_called_once_with(
            collection_name=vector_store.collection_name, data=mock_process_result
        )


def test_add_document_already_exists(vector_store):
    """Test adding a document that already exists."""
    pdf_path = "/path/to/test.pdf"

    # Mock the check for existing document - return a result indicating document exists
    vector_store.client.query.return_value = [{"source": "test"}]

    result = vector_store.add_document(pdf_path)

    assert result is True
    vector_store.client.query.assert_called_once()
    vector_store.client.insert.assert_not_called()


def test_retrieve(vector_store):
    """Test retrieving documents."""
    query_text = "test query"
    expected_results = [{"text": "result", "pk": 1}]
    vector_store.client.search.return_value = expected_results

    with mock.patch.object(vector_store, "_embed_fn") as mock_embed_fn:
        mock_embed_fn.encode_queries.return_value = [np.array([0.1, 0.2])]

        results = vector_store.retrieve(query_text)

        assert results == expected_results
        mock_embed_fn.encode_queries.assert_called_once_with([query_text])
        vector_store.client.search.assert_called_once()
