from unittest.mock import MagicMock, patch

import torch

from zotpilot.embeddings import (
    EmbeddingModel,
    embed_doc_chunks,
)


@patch("zotpilot.embeddings.SentenceTransformer")
@patch("zotpilot.embeddings.AutoTokenizer")
@patch("zotpilot.embeddings.get_device", return_value="cpu")
def test_embedding_model(mock_get_device, mock_tokenizer_cls, mock_sentence_transformer_cls):
    """Test basic functionality of EmbeddingModel"""
    # Setup mocks
    mock_model = MagicMock()
    mock_tokenizer = MagicMock()
    mock_sentence_transformer_cls.return_value = mock_model
    mock_tokenizer_cls.from_pretrained.return_value = mock_tokenizer
    mock_model.encode.return_value = torch.tensor([[0.1, 0.2, 0.3]])

    # Create model and test embedding
    model = EmbeddingModel()
    texts = ["Sample text"]
    result = model.embed_text(texts)

    assert isinstance(result, torch.Tensor)
    assert result.shape[0] == len(texts)  # one embedding per text


@patch("zotpilot.embeddings.SentenceTransformer")
@patch("zotpilot.embeddings.AutoTokenizer")
@patch("zotpilot.embeddings.get_device", return_value="cpu")
def test_embedding_model_string_vs_list(
    mock_get_device, mock_tokenizer_cls, mock_sentence_transformer_cls
):
    """Test that a string and a list with the same string return identical embeddings"""
    # Setup mocks
    mock_model = MagicMock()
    mock_tokenizer = MagicMock()
    mock_sentence_transformer_cls.return_value = mock_model
    mock_tokenizer_cls.from_pretrained.return_value = mock_tokenizer

    embedding_tensor = torch.tensor([[0.1, 0.2, 0.3]])
    mock_model.encode.return_value = embedding_tensor

    # Create model and test both string and list inputs
    model = EmbeddingModel()
    text = "Sample text"
    result_string = model.embed_text(text)
    result_list = model.embed_text([text])

    assert torch.equal(result_string, result_list)
    assert mock_model.encode.call_count == 2
    first_call_args = mock_model.encode.call_args_list[0][0]
    second_call_args = mock_model.encode.call_args_list[1][0]
    assert first_call_args[0] == [text]
    assert second_call_args[0] == [text]


def test_get_chunk_embeddings():
    """Test basic functionality of get_chunk_embeddings"""
    mock_chunk1 = MagicMock()
    mock_chunk1.text = "This is the first chunk"
    mock_chunk2 = MagicMock()
    mock_chunk2.text = "This is the second chunk"
    mock_chunks = [mock_chunk1, mock_chunk2]
    mock_model = MagicMock()
    mock_model.embed_text.return_value = torch.tensor([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
    texts, embeddings = embed_doc_chunks(mock_chunks, model=mock_model)

    assert len(texts) == len(mock_chunks)
    assert isinstance(embeddings, torch.Tensor)
    assert embeddings.shape[0] == len(mock_chunks)  # one embedding per chunk
    mock_model.embed_text.assert_called_once_with(
        [
            "This is the first chunk",
            "This is the second chunk",
        ]
    )
