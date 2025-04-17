"""Tests for the ingestion module."""

import datetime
from pathlib import Path
from unittest import mock

import numpy as np
import pytest
from docling.chunking import DocChunk
from docling.datamodel.document import DoclingDocument

from paperchat.core.ingestion import extract_page_numbers, parse_pdf, process_pdf_document


@pytest.fixture
def mock_docling_doc():
    """Return a mock DoclingDocument."""
    doc = mock.MagicMock(spec=DoclingDocument)
    return doc


@pytest.fixture
def mock_doc_converter():
    """Create a mock DocumentConverter."""
    with mock.patch("paperchat.core.ingestion.DocumentConverter") as mock_converter_cls:
        mock_converter = mock.MagicMock()
        mock_converter_cls.return_value = mock_converter

        mock_result = mock.MagicMock()
        mock_result.document = mock.MagicMock(spec=DoclingDocument)
        mock_converter.convert.return_value = mock_result

        yield mock_converter


def test_parse_pdf(mock_doc_converter):
    """Test that parse_pdf calls DocumentConverter correctly."""
    pdf_path = "/path/to/test.pdf"
    doc = parse_pdf(pdf_path)
    mock_doc_converter.convert.assert_called_once_with(str(Path(pdf_path).resolve()))

    assert doc is mock_doc_converter.convert.return_value.document


def test_extract_page_numbers():
    """Test extracting page numbers from a DocChunk."""
    # Create a mock DocChunk with page numbers 1, 2, 1
    chunk = mock.MagicMock(spec=DocChunk)
    chunk.meta = mock.MagicMock()
    doc_item1 = mock.MagicMock()
    doc_item1.prov = [mock.MagicMock(page_no=1)]
    doc_item2 = mock.MagicMock()
    doc_item2.prov = [mock.MagicMock(page_no=2)]
    doc_item3 = mock.MagicMock()
    doc_item3.prov = [mock.MagicMock(page_no=1)]

    chunk.meta.doc_items = [doc_item1, doc_item2, doc_item3]

    page_numbers = extract_page_numbers(chunk)

    assert page_numbers == [1, 2]


@mock.patch("paperchat.core.ingestion.parse_pdf")
@mock.patch("paperchat.core.ingestion.HybridChunker")
def test_process_pdf_document(mock_chunker_cls, mock_parse_pdf):
    """Test processing a PDF document."""
    pdf_path = "/path/to/test.pdf"
    mock_doc = mock.MagicMock()
    mock_parse_pdf.return_value = mock_doc

    mock_chunker = mock.MagicMock()
    mock_chunker_cls.return_value = mock_chunker

    # Create two mock chunks
    mock_chunk1 = mock.MagicMock(spec=DocChunk)
    mock_chunk1.text = "This is chunk 1"
    mock_chunk1.meta = mock.MagicMock()
    mock_chunk1.meta.doc_items = [mock.MagicMock()]
    mock_chunk1.meta.doc_items[0].prov = [mock.MagicMock(page_no=1)]
    mock_chunk1.meta.doc_items[0].label = "label1"
    mock_chunk1.meta.headings = ["Heading 1"]

    mock_chunk2 = mock.MagicMock(spec=DocChunk)
    mock_chunk2.text = "This is chunk 2"
    mock_chunk2.meta = mock.MagicMock()
    mock_chunk2.meta.doc_items = [mock.MagicMock()]
    mock_chunk2.meta.doc_items[0].prov = [mock.MagicMock(page_no=2)]
    mock_chunk2.meta.doc_items[0].label = "label2"
    mock_chunk2.meta.headings = ["Heading 2"]

    mock_chunks = [mock_chunk1, mock_chunk2]
    mock_chunker.chunk.return_value = mock_chunks

    mock_chunker.contextualize.side_effect = lambda chunk: f"Contextualized {chunk.text}"

    mock_embed_fn = mock.MagicMock()
    mock_embed_fn.encode_documents.return_value = [
        np.array([0.1, 0.2, 0.3]),
        np.array([0.4, 0.5, 0.6]),
    ]

    mock_datetime = datetime.datetime(2023, 1, 1, 12, 0, 0)
    with mock.patch("paperchat.core.ingestion.datetime.datetime") as mock_dt:
        mock_dt.now.return_value = mock_datetime

        result = process_pdf_document(pdf_path, embed_fn=mock_embed_fn)

    mock_parse_pdf.assert_called_once_with(Path(pdf_path))
    mock_chunker_cls.assert_called_once_with(max_tokens=1024, merge_peers=True)
    mock_chunker.chunk.assert_called_once_with(mock_doc)

    assert len(result) == 2

    assert result[0]["chunk_id"] == "chunk_00"
    assert result[0]["text"] == "This is chunk 1"
    assert result[0]["label"] == ["label1"]
    assert result[0]["page_numbers"] == [1]
    assert result[0]["headings"] == "heading 1"
    assert result[0]["source"] == Path(pdf_path).stem
    assert result[0]["timestamp"] == "2023-01-01T12:00:00"

    assert result[1]["chunk_id"] == "chunk_01"
    assert result[1]["text"] == "This is chunk 2"
    assert result[1]["label"] == ["label2"]
    assert result[1]["page_numbers"] == [2]
    assert result[1]["headings"] == "heading 2"
    assert result[1]["source"] == Path(pdf_path).stem
    assert result[1]["timestamp"] == "2023-01-01T12:00:00"

    # check that embed_fn was called with contextualized chunks
    mock_embed_fn.encode_documents.assert_called_once_with(
        ["Contextualized This is chunk 1", "Contextualized This is chunk 2"]
    )
