from pathlib import Path
from typing import Iterable, Iterator

import torch
from docling.chunking import DocChunk, HybridChunker
from docling.datamodel.document import DoclingDocument
from docling.document_converter import DocumentConverter
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer

from .settings import BATCH_SIZE, EMBEDDING_MODEL, get_device


def load_tokenizer(model_id: str | None = None) -> AutoTokenizer:
    """Load the tokenizer for embedding model.

    Args:
        model_id: HuggingFace model ID of tokenizer to load.
            If None, will use default model ID: "all-mpnet-base-v2"

    Returns:
        Loaded tokenizer
    """

    if model_id is None:
        model_id = EMBEDDING_MODEL

    return AutoTokenizer.from_pretrained(model_id)


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


def get_chunk_embeddings(
    chunks: Iterable[DocChunk],
    model_id: str | None = None,
    batch_size: int = BATCH_SIZE,
    device: str | None = None,
) -> tuple[list[str], torch.Tensor]:
    """Get embeddings for document chunks efficiently using batched processing.

    Args:
        chunks: Chunks to embed
        model_id: Model ID to use for embeddings, defaults to base model
        batch_size: Number of chunks to process at once
        device: Device to use. If None, uses value from settings.get_device()

    Returns:
        Tuple of (texts, embeddings) where:
            - texts is a list of chunk texts
            - embeddings is a 2D tensor of shape (n_chunks, embedding_dim)
    """
    if model_id is None:
        model_id = EMBEDDING_MODEL

    device = device or get_device()
    model = SentenceTransformer(model_id)
    model = model.to(device)
    chunk_list = list(chunks)
    texts = [chunk.text for chunk in chunk_list]
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_tensor=True,
        device=device,
    )

    return texts, embeddings
