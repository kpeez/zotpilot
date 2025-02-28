from typing import Iterable

import torch
from docling.chunking import DocChunk
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer

from .settings import BATCH_SIZE, EMBEDDING_MODEL, get_device


def load_tokenizer(model_id: str = EMBEDDING_MODEL) -> AutoTokenizer:
    """Load the tokenizer for embedding model.

    Args:
        model_id: HuggingFace model ID of tokenizer to load.
            If None, will use default model ID: "all-mpnet-base-v2"

    Returns:
        Loaded tokenizer
    """

    return AutoTokenizer.from_pretrained(model_id)


def get_text_embeddings(
    texts: str | list[str],
    model_id: str | None = None,
    batch_size: int = BATCH_SIZE,
    device: str | None = None,
) -> torch.Tensor:
    """Get embeddings for a list of texts.

    Args:
        texts: List of texts to embed
        model_id: Model ID to use for embeddings, defaults to base model
        batch_size: Number of texts to process at once
        device: Device to use. If None, uses value from settings.get_device()

    Returns:
        Tensor of shape (n_texts, embedding_dim) containing the embeddings
    """
    if isinstance(texts, str):
        texts = [texts]

    device = device or get_device()
    model_id = model_id or EMBEDDING_MODEL
    model = SentenceTransformer(model_id)
    model = model.to(device)

    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_tensor=True,
        device=device,
    )

    return embeddings


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
    chunk_list = list(chunks)
    texts = [chunk.text for chunk in chunk_list]
    model_id = model_id or EMBEDDING_MODEL
    embeddings = get_text_embeddings(texts, model_id, batch_size, device)
    return texts, embeddings
