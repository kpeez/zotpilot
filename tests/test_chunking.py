from pathlib import Path

import pytest
from docling.datamodel.document import DoclingDocument

from zotpilot.chunking import (
    chunk_document,
    create_chunker,
    parse_pdf,
    process_document,
)
from zotpilot.settings import EMBEDDING_MODEL

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
    chunker = create_chunker()
    return list(chunk_document(test_document, chunker=chunker))


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


def test_process_document():
    chunks = process_document(TEST_PDF_PATH, model_id=TEST_MODEL_ID)
    assert len(chunks) > 0
    assert all(len(chunk.text) > 20 for chunk in chunks)
