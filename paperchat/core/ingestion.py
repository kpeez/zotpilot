import datetime
from pathlib import Path
from typing import Any, Iterator

from docling.chunking import DocChunk, HybridChunker
from docling.datamodel.document import DoclingDocument
from docling.document_converter import DocumentConverter
from transformers import AutoTokenizer

from .embeddings import EmbeddingModel, embed_doc_chunks
from .vector_store import VectorStore


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


def chunk_document(
    document: DoclingDocument,
    tokenizer: AutoTokenizer | None = None,
    merge_peers: bool = True,
) -> Iterator[DocChunk]:
    """Chunk a document into semantic chunks.

    Args:
        document: Document to chunk
        tokenizer: Tokenizer to use, will load default if None
        merge_peers: Whether to merge peer chunks. Default is True.

    Returns:
        Iterator of document chunks
    """
    if tokenizer is None:
        embedding_model = EmbeddingModel()
        tokenizer = embedding_model.tokenizer

    max_tokens = tokenizer.model_max_length

    chunker = HybridChunker(
        tokenizer=tokenizer,
        max_tokens=max_tokens,
        merge_peers=merge_peers,
    )

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
    chunks = list(chunk_document(document, tokenizer=model.tokenizer))

    return chunks


def process_document(
    pdf_path: str | Path,
    embedding_model: EmbeddingModel,
    vector_store: VectorStore | None = None,
) -> dict[str, Any]:
    """Process a document to generate text and embeddings using a provided model.

    This function handles the full pipeline from raw PDF to text and embeddings
    and stores the results in the vector database.

    Args:
        pdf_path: Path to the PDF file
        embedding_model: EmbeddingModel instance to use for tokenization and embeddings
        vector_store: VectorStore instance to use for storage. If None, creates a new instance.

    Returns:
        Dictionary containing:
        - collection_name: Name derived from the PDF filename
        - chunk_texts: List of chunk texts
        - chunk_metadata: List of chunk metadata
        - chunk_embeddings: Tensor of chunk embeddings
    """
    pdf_path = Path(pdf_path)
    collection_name = pdf_path.stem

    chunks = get_pdf_chunks(pdf_path, model=embedding_model)
    chunk_texts, chunk_embeddings = embed_doc_chunks(chunks, model=embedding_model)
    chunk_metadata = [
        {
            "page": chunk.meta.doc_items[0].prov[0].page_no
            if chunk.meta.doc_items and chunk.meta.doc_items[0].prov
            else 0,
            "chunk_id": i,
            "headings": chunk.meta.headings,
            "source_file": str(pdf_path),
            "timestamp": datetime.datetime.now().isoformat(),
        }
        for i, chunk in enumerate(chunks)
    ]

    if vector_store is None:
        vector_store = VectorStore()

    vector_store.add_document(
        collection_name=collection_name,
        chunk_texts=chunk_texts,
        chunk_embeddings=chunk_embeddings,
        chunk_metadata=chunk_metadata,
    )

    return {
        "collection_name": collection_name,
        "chunk_texts": chunk_texts,
        "chunk_metadata": chunk_metadata,
        "chunk_embeddings": chunk_embeddings,
    }
