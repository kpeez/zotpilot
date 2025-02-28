from unittest.mock import MagicMock, patch

import torch

from zotpilot.embeddings import (
    get_chunk_embeddings,
    get_text_embeddings,
)


@patch("zotpilot.embeddings.SentenceTransformer")
@patch("zotpilot.embeddings.get_device", return_value="cpu")
def test_get_text_embeddings(mock_get_device, mock_sentence_transformer_cls):
    """Test basic functionality of get_text_embeddings"""
    mock_model = MagicMock()
    mock_sentence_transformer_cls.return_value = mock_model
    mock_model.to.return_value = mock_model
    mock_model.encode.return_value = torch.tensor([[0.1, 0.2, 0.3]])
    texts = ["Sample text"]
    result = get_text_embeddings(texts)
    assert isinstance(result, torch.Tensor)
    assert result.shape[0] == len(texts)  # one embedding per text


@patch("zotpilot.embeddings.SentenceTransformer")
@patch("zotpilot.embeddings.get_device", return_value="cpu")
def test_get_text_embeddings_string_vs_list(mock_get_device, mock_sentence_transformer_cls):
    """Test that a string and a list with the same string return identical embeddings"""
    mock_model = MagicMock()
    mock_sentence_transformer_cls.return_value = mock_model
    mock_model.to.return_value = mock_model

    embedding_tensor = torch.tensor([[0.1, 0.2, 0.3]])
    mock_model.encode.return_value = embedding_tensor
    text = "Sample text"
    result_string = get_text_embeddings(text)
    result_list = get_text_embeddings([text])

    assert torch.equal(result_string, result_list)
    assert mock_model.encode.call_count == 2
    first_call_args = mock_model.encode.call_args_list[0][0]
    second_call_args = mock_model.encode.call_args_list[1][0]
    assert first_call_args[0] == [text]
    assert second_call_args[0] == [text]


@patch("zotpilot.embeddings.get_text_embeddings")
def test_get_chunk_embeddings(mock_get_text_embeddings):
    """Test basic functionality of get_chunk_embeddings"""
    # Create mock chunks
    mock_chunk1 = MagicMock()
    mock_chunk1.text = "This is the first chunk"
    mock_chunk2 = MagicMock()
    mock_chunk2.text = "This is the second chunk"
    mock_chunks = [mock_chunk1, mock_chunk2]
    mock_get_text_embeddings.return_value = torch.tensor([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
    texts, embeddings = get_chunk_embeddings(mock_chunks)
    assert len(texts) == len(mock_chunks)
    assert isinstance(embeddings, torch.Tensor)
    assert embeddings.shape[0] == len(mock_chunks)  # one embedding per chunk
