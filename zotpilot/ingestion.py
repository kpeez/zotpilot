from pathlib import Path
from typing import Iterator

from docling.chunking import DocChunk, HybridChunker
from docling.datamodel.document import DoclingDocument
from docling.document_converter import DocumentConverter
from transformers import AutoTokenizer

from .embeddings import load_tokenizer


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


def create_chunker(
    tokenizer: AutoTokenizer | None = None,
    max_tokens: int = 512,  # Ensure this matches your model's limit
    merge_peers: bool = True,
) -> HybridChunker:
    """Create a document chunker with specified settings.

    Args:
        tokenizer: Tokenizer to use, will load default if None
        max_tokens: Maximum tokens per chunk. Default is 384 to ensure chunks plus
                   metadata stay under model's 512 token limit
        merge_peers: Whether to merge peer chunks

    Returns:
        Configured chunker
    """
    if tokenizer is None:
        tokenizer = load_tokenizer()

    # Add validation
    if max_tokens > 512:
        raise ValueError("Chunk size cannot exceed 512 tokens for this model")

    return HybridChunker(
        tokenizer=tokenizer,
        max_tokens=max_tokens,
        merge_peers=merge_peers,
    )


def chunk_document(
    document: DoclingDocument, chunker: HybridChunker | None = None
) -> Iterator[DocChunk]:
    """Chunk a document into semantic chunks.

    Args:
        document: Document to chunk
        chunker: Chunker to use, will create default if None

    Returns:
        Iterator of document chunks
    """
    if chunker is None:
        chunker = create_chunker()

    return chunker.chunk(document)


def process_document(pdf_path: str | Path, model_id: str | None = None) -> list[DocChunk]:
    """Process a PDF document into chunks.

    Args:
        pdf_path: Path to the PDF file (can be string or Path object)
        model_id: Model ID to use for tokenization

    Returns:
        List of document chunks
    """
    tokenizer = load_tokenizer(model_id=model_id)
    document = parse_pdf(pdf_path)
    chunker = create_chunker(tokenizer=tokenizer)
    chunks = list(chunk_document(document, chunker=chunker))

    return chunks
