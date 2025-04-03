from pathlib import Path
from typing import Any, Iterator

from docling.chunking import DocChunk, HybridChunker
from docling.datamodel.document import DoclingDocument
from docling.document_converter import DocumentConverter
from transformers import AutoTokenizer

from .embeddings import EmbeddingModel, get_chunk_embeddings


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
    max_tokens: int = 512,
    merge_peers: bool = True,
) -> HybridChunker:
    """Create a document chunker with specified settings.

    Args:
        tokenizer: Tokenizer to use, will load default if None
        max_tokens: Maximum tokens per chunk
        merge_peers: Whether to merge peer chunks

    Returns:
        Configured chunker
    """
    if tokenizer is None:
        embedding_model = EmbeddingModel()
        tokenizer = embedding_model.tokenizer

    if max_tokens > tokenizer.model_max_length:
        raise ValueError(
            f"Chunk size cannot exceed {tokenizer.model_max_length} tokens for this model"
        )

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


def get_pdf_chunks(
    pdf_path: str | Path,
    model: EmbeddingModel,
) -> list[DocChunk]:
    """Get chunks from a PDF document.

    Args:
        pdf_path: Path to the PDF file (can be string or Path object)
        model: EmbeddingModel instance to use for tokenization

    Returns:
        List of document chunks
    """
    document = parse_pdf(pdf_path)
    chunker = create_chunker(tokenizer=model.tokenizer)
    chunks = list(chunk_document(document, chunker=chunker))

    return chunks


def process_document(
    pdf_path: str | Path, model_id: str | None = None, embedding_model: Any = None
) -> dict[str, Any]:
    """Process a document to generate text and embeddings.

    This function handles the full pipeline from raw PDF to text and embeddings.

    Args:
        pdf_path: Path to the PDF file
        model_id: Model ID to use for tokenization and embeddings (ignored if model is provided)
        embedding_model: Optional pre-initialized EmbeddingModel instance

    Returns:
        Dictionary containing:
        - collection_name: Name derived from the PDF filename
        - chunk_texts: List of chunk texts
        - chunk_metadata: List of chunk metadata
        - chunk_embeddings: Tensor of chunk embeddings
    """
    collection_name = Path(pdf_path).stem
    if embedding_model is None:
        from .embeddings import EmbeddingModel

        embedding_model = EmbeddingModel(model_id=model_id)

    chunks = get_pdf_chunks(pdf_path, model=embedding_model)
    chunk_texts, chunk_embeddings = get_chunk_embeddings(chunks, model=embedding_model)
    chunk_metadata = [
        {
            "page": chunk.meta.doc_items[0].prov[0].page_no
            if chunk.meta.doc_items and chunk.meta.doc_items[0].prov
            else 0,
            "chunk_id": i,
        }
        for i, chunk in enumerate(chunks)
    ]

    return {
        "collection_name": collection_name,
        "chunk_texts": chunk_texts,
        "chunk_metadata": chunk_metadata,
        "chunk_embeddings": chunk_embeddings,
    }
