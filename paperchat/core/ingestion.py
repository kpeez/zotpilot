import datetime
from pathlib import Path
from typing import Any

from docling.chunking import DocChunk, HybridChunker
from docling.datamodel.document import DoclingDocument
from docling.document_converter import DocumentConverter
from pymilvus import model


def parse_pdf(pdf_path: str | Path) -> DoclingDocument:
    """Convert a PDF file to a Docling document.

    Args:
        pdf_path: Path to the PDF file (can be string or Path object)

    Returns:
        Parsed Docling document
    """
    converter = DocumentConverter()
    pdf_path = Path(pdf_path).resolve()
    result = converter.convert(str(pdf_path))

    return result.document


def extract_page_numbers(chunk: DocChunk) -> list[int]:
    """Extract page numbers from a DocChunk.

    Args:
        chunk: DocChunk to extract page numbers from

    Returns:
        List of page numbers
    """
    page_numbers = set()
    for item in chunk.meta.doc_items:
        for prov in item.prov:
            page_numbers.add(prov.page_no)

    return sorted(page_numbers)


def process_pdf_document(
    pdf_path: str | Path,
    embed_fn: Any | None = None,
    max_tokens_per_chunk: int = 1024,
) -> list[dict[str, Any]]:
    """Process a PDF document into a list of dictionaries ready for vector database insertion.

    This function handles the document processing pipeline and returns data in a format
    ready to be directly inserted into a vector database.

    Args:
        pdf_path: Path to the PDF file
        embed_fn: Function to use for embedding generation, defaults to Milvus DefaultEmbeddingFunction
        max_tokens_per_chunk: Maximum number of tokens per chunk

    Returns:
        List of dictionaries with fields matching the Milvus schema, ready for direct insertion
    """
    if embed_fn is None:
        embed_fn = model.DefaultEmbeddingFunction()

    pdf_file = Path(pdf_path)
    pdf_doc = parse_pdf(pdf_file)
    chunker = HybridChunker(max_tokens=max_tokens_per_chunk, merge_peers=True)
    pdf_chunks = list(chunker.chunk(pdf_doc))

    texts = [chunker.contextualize(chunk) for chunk in pdf_chunks]
    embeddings = embed_fn.encode_documents(texts)

    data_entries = []
    for i, (chunk, embedding) in enumerate(zip(pdf_chunks, embeddings, strict=True)):
        data_entry = {
            "chunk_id": f"chunk_{i:02d}",
            "embedding": embedding,
            "text": chunk.text,
            "label": [item.label for item in chunk.meta.doc_items or []],
            "source": pdf_file.stem,
            "page_numbers": extract_page_numbers(chunk),
            "headings": ", ".join(chunk.meta.headings).lower() if chunk.meta.headings else "",
            "timestamp": datetime.datetime.now().isoformat(),
        }
        data_entries.append(data_entry)

    return data_entries
