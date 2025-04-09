from pathlib import Path

import pytest
import torch
from docling.datamodel.document import DoclingDocument

from paperchat.embeddings import EmbeddingModel
from paperchat.ingestion import chunk_document, get_pdf_chunks, parse_pdf, process_document
from paperchat.utils.config import EMBEDDING_MODEL

# Ignore the deprecation warnings from docling
pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")


TEST_PDF_PATH = Path("./tests/data/lewis_etal_2021_rag.pdf")
TEST_MODEL_ID = EMBEDDING_MODEL


@pytest.fixture(scope="module")
def test_document():
    """Cache the parsed document for all tests"""
    return parse_pdf(TEST_PDF_PATH)


@pytest.fixture(scope="module")
def test_chunks(test_document):
    """Cache the chunks for all tests"""
    model = EmbeddingModel(model_id=TEST_MODEL_ID)
    return list(chunk_document(test_document, tokenizer=model.tokenizer))


def test_parse_pdf_returns_docling_document(test_document):
    """Test that parse_pdf returns a valid DoclingDocument instance"""
    assert isinstance(test_document, DoclingDocument)


def test_parse_pdf_contains_pages(test_document):
    """Test that parsed document contains at least one page"""
    assert hasattr(test_document, "pages")
    assert len(test_document.pages) > 0
    first_page_num = min(test_document.pages.keys())
    assert test_document.pages[first_page_num] is not None


def test_parse_pdf_document_structure(test_document):
    """Test that parsed document has expected high-level structure"""
    assert hasattr(test_document, "texts") or hasattr(test_document, "body"), (
        "Document must have either texts or body content"
    )
    if hasattr(test_document, "texts"):
        assert isinstance(test_document.texts, (list, tuple)), "texts should be a sequence"
    if hasattr(test_document, "body"):
        assert test_document.body is not None, "body should not be None if present"


def test_parse_invalid_file():
    """Test that parse_pdf handles invalid files appropriately"""
    with pytest.raises(FileNotFoundError):
        parse_pdf("nonexistent.pdf")


def test_get_pdf_chunks():
    """Test that get_pdf_chunks returns valid chunks"""
    model = EmbeddingModel(model_id=TEST_MODEL_ID)
    chunks = get_pdf_chunks(TEST_PDF_PATH, model=model)
    assert len(chunks) > 0
    assert all(len(chunk.text) > 20 for chunk in chunks)


def test_process_document_returns_expected_structure():
    """Test that process_document returns a dictionary with the expected structure"""
    result = process_document(TEST_PDF_PATH, model_id=TEST_MODEL_ID)

    assert isinstance(result, dict)
    assert "collection_name" in result
    assert "chunk_texts" in result
    assert "chunk_metadata" in result
    assert "chunk_embeddings" in result
    assert result["collection_name"] == TEST_PDF_PATH.stem

    assert isinstance(result["chunk_texts"], list)
    assert len(result["chunk_texts"]) > 0
    assert all(isinstance(text, str) for text in result["chunk_texts"])

    assert isinstance(result["chunk_metadata"], list)
    assert len(result["chunk_metadata"]) == len(result["chunk_texts"])
    assert all(isinstance(meta, dict) for meta in result["chunk_metadata"])
    assert all("page" in meta for meta in result["chunk_metadata"])
    assert all("chunk_id" in meta for meta in result["chunk_metadata"])
    assert isinstance(result["chunk_embeddings"], torch.Tensor)
    assert result["chunk_embeddings"].shape[0] == len(result["chunk_texts"])


def test_process_document_consistency():
    """Test that process_document produces consistent results between texts and embeddings"""
    result = process_document(TEST_PDF_PATH, model_id=TEST_MODEL_ID)
    assert len(result["chunk_texts"]) == result["chunk_embeddings"].shape[0]
    assert len(result["chunk_metadata"]) == len(result["chunk_texts"])
    chunk_ids = [meta["chunk_id"] for meta in result["chunk_metadata"]]
    assert chunk_ids == list(range(len(chunk_ids)))
