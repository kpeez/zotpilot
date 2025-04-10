"""Tests for the API key management utilities."""

import os
from unittest import mock

import keyring

from paperchat.utils.api_keys import (
    _get_keyring_username,
    _validate_key,
    get_api_key,
    get_available_providers,
    get_provider_display_name,
    mask_api_key,
    remove_api_key,
    set_api_key,
)


def test_get_keyring_username():
    """Test the username generation for keyring storage."""
    assert _get_keyring_username("openai") == "openai_api_key"
    assert _get_keyring_username("ANTHROPIC") == "anthropic_api_key"
    assert _get_keyring_username("custom") == "custom_api_key"


def test_validate_key():
    """Test API key validation logic."""
    is_valid, error = _validate_key("openai", "sk-abcdefghijklmnop")
    assert is_valid
    assert error is None

    is_valid, error = _validate_key("anthropic", "sk-ant-api12345")
    assert is_valid
    assert error is None

    is_valid, error = _validate_key("openai", "")
    assert not is_valid
    assert "cannot be empty" in error

    is_valid, error = _validate_key("anthropic", "not-a-valid-key")
    assert not is_valid
    assert "should start with" in error

    is_valid, error = _validate_key("custom", "any-key-works")
    assert is_valid
    assert error is None


@mock.patch.dict(os.environ, {"OPENAI_API_KEY": "env-openai-key", "ANTHROPIC_API_KEY": ""})
def test_get_api_key_from_env():
    """Test retrieving API key from environment variables."""
    assert get_api_key("openai") == "env-openai-key"


@mock.patch.dict(os.environ, {"OPENAI_API_KEY": "", "ANTHROPIC_API_KEY": ""})
@mock.patch("keyring.get_password")
def test_get_api_key_from_keyring(mock_get_password):
    """Test retrieving API key from keyring when not in env vars."""
    mock_get_password.return_value = "keyring-anthropic-key"
    assert get_api_key("anthropic") == "keyring-anthropic-key"
    mock_get_password.assert_called_once()


@mock.patch.dict(os.environ, {"OPENAI_API_KEY": "", "ANTHROPIC_API_KEY": ""})
@mock.patch("keyring.get_password")
def test_get_api_key_not_found(mock_get_password):
    """Test behavior when API key is not found anywhere."""
    mock_get_password.return_value = None
    assert get_api_key("openai") is None


@mock.patch.dict(os.environ, {"OPENAI_API_KEY": "", "ANTHROPIC_API_KEY": ""})
@mock.patch("keyring.get_password")
def test_get_api_key_keyring_exception(mock_get_password):
    """Test handling keyring exception gracefully."""
    mock_get_password.side_effect = Exception("Keyring error")
    assert get_api_key("openai") is None


@mock.patch("keyring.set_password")
def test_set_api_key_success(mock_set_password):
    """Test setting an API key successfully."""
    success, error = set_api_key("openai", "sk-validkey12345")
    assert success
    assert error is None
    mock_set_password.assert_called_once()


@mock.patch("keyring.set_password")
def test_set_api_key_invalid(mock_set_password):
    """Test setting an invalid API key."""
    success, error = set_api_key("anthropic", "invalid-key")
    assert not success
    assert "should start with" in error
    mock_set_password.assert_not_called()


@mock.patch("keyring.set_password")
def test_set_api_key_exception(mock_set_password):
    """Test handling exception when setting API key."""
    mock_set_password.side_effect = Exception("Set password error")
    success, error = set_api_key("openai", "sk-validkey12345")
    assert not success
    assert "Error saving" in error


@mock.patch("keyring.get_password")
@mock.patch("keyring.delete_password")
def test_remove_api_key_success(mock_delete, mock_get):
    """Test removing an API key successfully."""
    mock_get.return_value = "some-key"
    success, message = remove_api_key("openai")
    assert success
    assert "successfully removed" in message
    mock_delete.assert_called_once()


@mock.patch("keyring.get_password")
@mock.patch("keyring.delete_password")
def test_remove_api_key_not_found(mock_delete, mock_get):
    """Test removing a non-existent API key."""
    mock_get.return_value = None
    success, message = remove_api_key("openai")
    assert success
    assert "No OpenAI key found" in message
    mock_delete.assert_not_called()


@mock.patch("keyring.get_password")
@mock.patch("keyring.delete_password")
def test_remove_api_key_delete_error(mock_delete, mock_get):
    """Test handling error when deleting API key."""
    mock_get.return_value = "some-key"
    mock_delete.side_effect = keyring.errors.PasswordDeleteError()
    success, message = remove_api_key("openai")
    assert success
    assert "No OpenAI key found" in message


@mock.patch("keyring.get_password")
@mock.patch("keyring.delete_password")
def test_remove_api_key_other_exception(mock_delete, mock_get):
    """Test handling other exceptions when removing API key."""
    mock_get.return_value = "some-key"
    mock_delete.side_effect = Exception("Generic error")
    success, message = remove_api_key("openai")
    assert not success
    assert "Error removing" in message


def test_get_provider_display_name():
    """Test getting display names for providers."""
    assert get_provider_display_name("openai") == "OpenAI"
    assert get_provider_display_name("anthropic") == "Anthropic"
    assert get_provider_display_name("custom") == "Custom"


def test_get_available_providers():
    """Test getting list of available providers."""
    providers = get_available_providers()
    assert "openai" in providers
    assert "anthropic" in providers
    assert isinstance(providers, list)


def test_mask_api_key():
    """Test masking API keys for display."""
    masked = mask_api_key("sk-1234567890abcdef")
    assert masked.startswith("sk-1")
    assert masked.endswith("cdef")
    assert "*" in masked

    assert mask_api_key("short") == "****"
    assert mask_api_key("") == "****"
    assert mask_api_key("12345678").startswith("1234")
    assert mask_api_key("12345678").endswith("5678")
