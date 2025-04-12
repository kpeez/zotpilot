"""Tests for the vector store module."""

import os
import tempfile
from pathlib import Path
from unittest import mock

import numpy as np
import pytest
import torch

from paperchat.core.vector_store import VectorStore


@pytest.fixture
def temp_db_dir():
    """Create a temporary directory for the test database."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def mock_chroma_client():
    """Create a mock ChromaDB client."""
    with mock.patch("chromadb.PersistentClient") as mock_client_class:
        mock_client = mock.MagicMock()
        mock_client_class.return_value = mock_client
        yield mock_client


@pytest.fixture
def vector_store(temp_db_dir, mock_chroma_client):
    """Create a VectorStore instance with a mocked ChromaDB client."""
    store = VectorStore(persist_directory=temp_db_dir)
    store.client = mock_chroma_client
    yield store


def test_init_creates_directory(temp_db_dir):
    """Test that initialization creates the directory if it doesn't exist."""
    if os.path.exists(temp_db_dir):
        os.rmdir(temp_db_dir)

    with mock.patch("chromadb.PersistentClient"):
        VectorStore(persist_directory=temp_db_dir)
        assert os.path.exists(temp_db_dir)


def test_init_uses_default_directory():
    """Test that initialization uses the default directory when none is provided."""
    with mock.patch("pathlib.Path.home") as mock_home:
        mock_home.return_value = Path("/mock/home")
        with mock.patch("os.makedirs") as mock_makedirs, mock.patch("chromadb.PersistentClient"):
            store = VectorStore()
            assert store.persist_directory == "/mock/home/.paperchat/vectordb"
            mock_makedirs.assert_called_once_with("/mock/home/.paperchat/vectordb", exist_ok=True)


def test_list_collections(vector_store):
    """Test listing collections."""
    mock_collection1 = mock.MagicMock()
    mock_collection1.name = "collection1"
    mock_collection2 = mock.MagicMock()
    mock_collection2.name = "collection2"
    vector_store.client.list_collections.return_value = [mock_collection1, mock_collection2]

    collections = vector_store.list_collections()
    assert collections == ["collection1", "collection2"]
    vector_store.client.list_collections.assert_called_once()


def test_create_collection(vector_store):
    """Test creating a collection."""
    vector_store.create_collection("test_collection")
    vector_store.client.get_or_create_collection.assert_called_once_with(name="test_collection")


def test_delete_collection(vector_store):
    """Test deleting a collection."""
    vector_store.delete_collection("test_collection")
    vector_store.client.delete_collection.assert_called_once_with(name="test_collection")


def test_add_document_with_torch_tensor(vector_store):
    """Test adding documents with a torch tensor."""
    collection_name = "test_collection"
    mock_collection = mock.MagicMock()
    vector_store.client.get_or_create_collection.return_value = mock_collection

    chunk_texts = ["text1", "text2"]
    chunk_embeddings = torch.tensor([[0.1, 0.2], [0.3, 0.4]])
    chunk_metadata = [{"page": 1}, {"page": 2}]

    vector_store.add_document(
        collection_name=collection_name,
        chunk_texts=chunk_texts,
        chunk_embeddings=chunk_embeddings,
        chunk_metadata=chunk_metadata,
    )

    vector_store.client.get_or_create_collection.assert_called_once_with(name=collection_name)
    mock_collection.add.assert_called_once()
    args, kwargs = mock_collection.add.call_args

    assert isinstance(kwargs["embeddings"], np.ndarray)
    assert kwargs["ids"] == ["test_collection_0", "test_collection_1"]
    assert kwargs["documents"] == chunk_texts
    assert kwargs["metadatas"] == chunk_metadata


def test_add_document_with_numpy_array(vector_store):
    """Test adding documents with a numpy array."""
    collection_name = "test_collection"
    mock_collection = mock.MagicMock()
    vector_store.client.get_or_create_collection.return_value = mock_collection

    chunk_texts = ["text1", "text2"]
    chunk_embeddings = np.array([[0.1, 0.2], [0.3, 0.4]])
    chunk_metadata = [{"page": 1}, {"page": 2}]
    ids = ["id1", "id2"]

    vector_store.add_document(
        collection_name=collection_name,
        chunk_texts=chunk_texts,
        chunk_embeddings=chunk_embeddings,
        chunk_metadata=chunk_metadata,
        ids=ids,
    )

    vector_store.client.get_or_create_collection.assert_called_once_with(name=collection_name)
    mock_collection.add.assert_called_once()
    args, kwargs = mock_collection.add.call_args

    assert isinstance(kwargs["embeddings"], np.ndarray)
    assert kwargs["ids"] == ids
    assert kwargs["documents"] == chunk_texts
    assert kwargs["metadatas"] == chunk_metadata


def test_update_embeddings(vector_store):
    """Test updating embeddings for existing documents."""
    collection_name = "test_collection"
    mock_collection = mock.MagicMock()
    vector_store.client.get_collection.return_value = mock_collection

    ids = ["id1", "id2"]
    embeddings = torch.tensor([[0.1, 0.2], [0.3, 0.4]])

    vector_store.update_embeddings(
        collection_name=collection_name,
        ids=ids,
        embeddings=embeddings,
    )

    vector_store.client.get_collection.assert_called_once_with(name=collection_name)
    mock_collection.update.assert_called_once()
    args, kwargs = mock_collection.update.call_args

    assert isinstance(kwargs["embeddings"], np.ndarray)
    assert kwargs["ids"] == ids


def test_reembed_collection(vector_store):
    """Test re-embedding an entire collection."""
    collection_name = "test_collection"
    mock_collection = mock.MagicMock()
    vector_store.client.get_collection.return_value = mock_collection

    mock_collection.get.return_value = {
        "ids": ["id1", "id2", "id3", "id4", "id5"],
        "documents": ["text1", "text2", "text3", "text4", "text5"],
        "metadatas": [{"page": i} for i in range(1, 6)],
    }

    mock_embedding_model = mock.MagicMock()
    mock_embedding_model.embed_text.return_value = torch.tensor([[0.1, 0.2], [0.3, 0.4]])

    vector_store.update_embeddings = mock.MagicMock()

    vector_store.reembed_collection(
        collection_name=collection_name,
        embedding_model=mock_embedding_model,
        batch_size=2,
    )

    vector_store.client.get_collection.assert_called_once_with(name=collection_name)
    mock_collection.get.assert_called_once()

    assert mock_embedding_model.embed_text.call_count == 3
    assert vector_store.update_embeddings.call_count == 3

    args, kwargs = vector_store.update_embeddings.call_args_list[0]
    assert kwargs["collection_name"] == collection_name
    assert kwargs["ids"] == ["id1", "id2"]

    args, kwargs = vector_store.update_embeddings.call_args_list[1]
    assert kwargs["ids"] == ["id3", "id4"]

    args, kwargs = vector_store.update_embeddings.call_args_list[2]
    assert kwargs["ids"] == ["id5"]


def test_search(vector_store):
    """Test searching for similar documents."""
    query_embedding = torch.tensor([0.1, 0.2])
    collection_name = "test_collection"

    mock_collection = mock.MagicMock()
    vector_store.client.get_collection.return_value = mock_collection

    mock_collection.query.return_value = {
        "ids": [["id1", "id2"]],
        "documents": [["text1", "text2"]],
        "metadatas": [[{"page": 1}, {"page": 2}]],
        "distances": [[0.2, 0.4]],  # 1-distance is similarity (0.8, 0.6)
    }

    results = vector_store.search(
        query_embedding=query_embedding,
        collection_names=collection_name,
        threshold=0.7,
    )

    vector_store.client.get_collection.assert_called_once_with(name=collection_name)
    mock_collection.query.assert_called_once()

    assert results[collection_name]["documents"] == ["text1"]
    assert results[collection_name]["ids"] == ["id1"]
    assert results[collection_name]["metadatas"] == [{"page": 1}]
    assert results[collection_name]["distances"] == [0.2]


def test_search_multi_collection(vector_store):
    """Test searching across multiple collections."""
    query_embedding = torch.tensor([0.1, 0.2])
    collection_names = ["collection1", "collection2"]

    mock_collection1 = mock.MagicMock()
    mock_collection2 = mock.MagicMock()

    def mock_get_collection(name):
        if name == "collection1":
            return mock_collection1
        elif name == "collection2":
            return mock_collection2
        raise ValueError(f"Unknown collection {name}")

    vector_store.client.get_collection.side_effect = mock_get_collection

    mock_collection1.query.return_value = {
        "ids": [["id1"]],
        "documents": [["text1"]],
        "metadatas": [[{"page": 1}]],
        "distances": [[0.1]],  # similarity 0.9
    }

    mock_collection2.query.return_value = {
        "ids": [["id2"]],
        "documents": [["text2"]],
        "metadatas": [[{"page": 2}]],
        "distances": [[0.3]],  # similarity 0.7
    }

    results = vector_store.search(
        query_embedding=query_embedding,
        collection_names=collection_names,
    )

    assert vector_store.client.get_collection.call_count == 2
    assert mock_collection1.query.called
    assert mock_collection2.query.called

    assert set(results.keys()) == set(collection_names)
    assert results["collection1"]["documents"] == ["text1"]
    assert results["collection2"]["documents"] == ["text2"]


def test_get_collection_info(vector_store):
    """Test getting information about a collection."""
    collection_name = "test_collection"
    mock_collection = mock.MagicMock()
    mock_collection.count.return_value = 5
    vector_store.client.get_collection.return_value = mock_collection
    info = vector_store.get_collection_info(collection_name)
    vector_store.client.get_collection.assert_called_once_with(name=collection_name)
    mock_collection.count.assert_called_once()
    assert info == {"name": collection_name, "count": 5}


def test_get_document_texts(vector_store):
    """Test getting document texts and metadata from a collection."""
    collection_name = "test_collection"
    mock_collection = mock.MagicMock()
    mock_result = {
        "ids": ["id1", "id2"],
        "documents": ["text1", "text2"],
        "metadatas": [{"page": 1}, {"page": 2}],
    }
    mock_collection.get.return_value = mock_result
    vector_store.client.get_collection.return_value = mock_collection

    result = vector_store.get_document_texts(collection_name)

    vector_store.client.get_collection.assert_called_once_with(name=collection_name)
    mock_collection.get.assert_called_once()
    assert result == mock_result
