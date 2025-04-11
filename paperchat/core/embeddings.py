from typing import Iterable

import torch
from docling.chunking import DocChunk
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer

from ..utils.config import BATCH_SIZE, EMBEDDING_MODEL, get_device


class EmbeddingModel:
    """Wrapper for embedding model that handles tokenization and embedding.

    This class provides a unified interface for working with embedding models,
    handling both tokenization and embedding generation.
    """

    def __init__(
        self,
        model_id: str | None = None,
        device: str | None = None,
        batch_size: int = BATCH_SIZE,
    ):
        """Initialize the embedding model.

        Args:
            model_id: HuggingFace model ID to use. If None, uses default from settings.
            device: Device to use for computation. If None, uses value from settings.
            batch_size: Default batch size for embedding operations.
        """
        self.model_id = model_id or EMBEDDING_MODEL
        self.device = device or get_device()
        self.batch_size = batch_size
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self._model = SentenceTransformer(self.model_id, device=self.device)

    @property
    def tokenizer(self) -> AutoTokenizer:
        """Get the tokenizer for this embedding model."""
        return self._tokenizer

    @property
    def model(self) -> SentenceTransformer:
        """Get the model for this embedding model."""
        return self._model

    def embed_text(
        self,
        texts: str | list[str],
        batch_size: int | None = None,
    ) -> torch.Tensor:
        """Embed text or list of texts.

        Args:
            texts: Text string or list of texts to embed
            batch_size: Optional batch size override

        Returns:
            Tensor of shape (n_texts, embedding_dim) containing the embeddings
        """
        if isinstance(texts, str):
            texts = [texts]

        embeddings = self.model.encode(
            texts,
            batch_size=batch_size or self.batch_size,
            show_progress_bar=True,
            convert_to_tensor=True,
            device=self.device,
        )

        return embeddings


def embed_doc_chunks(
    chunks: Iterable[DocChunk],
    model: EmbeddingModel,
) -> tuple[list[str], torch.Tensor]:
    """Get embeddings for document chunks efficiently using batched processing.

    Args:
        chunks: Chunks to embed
        model: EmbeddingModel instance to use

    Returns:
        Tuple of (texts, embeddings) where:
            - texts is a list of chunk texts
            - embeddings is a 2D tensor of shape (n_chunks, embedding_dim)
    """
    chunk_list = list(chunks)
    texts = [chunk.text for chunk in chunk_list]
    embeddings = model.embed_text(texts)

    return texts, embeddings
