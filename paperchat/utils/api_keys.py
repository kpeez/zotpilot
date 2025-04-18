"""
API Key Management Utilities using System Keyring.
"""

import os
from typing import Final

import keyring
import keyring.errors

_KEYRING_SERVICE_NAME: Final[str] = "paperchat"
_PROVIDER_CONFIG: Final[dict[str, dict[str, str | None]]] = {
    "openai": {"env_var": "OPENAI_API_KEY", "prefix": "sk-", "name": "OpenAI"},
    "gemini": {"env_var": "GEMINI_API_KEY", "prefix": None, "name": "Gemini"},
    "anthropic": {"env_var": "ANTHROPIC_API_KEY", "prefix": "sk-ant-", "name": "Anthropic"},
}


def _get_keyring_username(provider: str) -> str:
    """Generates the username for storing the provider's key in keyring."""
    return f"{provider.lower()}_api_key"


def _validate_key(provider: str, key: str) -> tuple[bool, str | None]:
    """
    Performs basic validation on the API key format.

    Args:
        provider: The name of the provider (e.g., 'openai', 'anthropic')
        key: The API key to validate

    Returns:
        Tuple of (is_valid, error_message)
        - is_valid: True if the key is valid, False otherwise
        - error_message: None if valid, error description if invalid
    """
    config = _PROVIDER_CONFIG.get(provider.lower())
    if not config:
        return True, None  # Cannot validate unknown providers

    prefix = config.get("prefix")
    provider_name = config.get("name", provider.capitalize())

    if not key:
        return False, "API key cannot be empty."

    if prefix and not key.startswith(prefix):
        return False, f"Invalid format. {provider_name} API keys should start with '{prefix}'"

    return True, None


def get_api_key(provider: str) -> str | None:
    """
    Gets the API key for the specified provider.

    Checks environment variables first, then the system keyring.

    Args:
        provider: The name of the provider (e.g., 'openai', 'anthropic'). Case-insensitive.

    Returns:
        The API key string, or None if not found or if keyring access fails.
    """
    provider_lower = provider.lower()
    config = _PROVIDER_CONFIG.get(provider_lower)
    env_var_name = config.get("env_var") if config else None

    if env_var_name:
        api_key = os.environ.get(env_var_name)
        if api_key:
            return api_key

    try:
        keyring_username = _get_keyring_username(provider_lower)
        api_key = keyring.get_password(_KEYRING_SERVICE_NAME, keyring_username)
        if api_key:
            return api_key
    except Exception:
        # Just return None to indicate key not found
        pass

    return None


def set_api_key(provider: str, api_key: str) -> tuple[bool, str | None]:
    """
    Securely saves the API key for the specified provider using the system keyring.

    Args:
        provider: The name of the provider (e.g., 'openai'). Case-insensitive.
        api_key: The API key string to save.

    Returns:
        Tuple of (success, error_message)
        - success: True if saving was successful, False otherwise
        - error_message: None if successful, error description if failed
    """
    provider_lower = provider.lower()
    is_valid, error_msg = _validate_key(provider_lower, api_key)

    if not is_valid:
        return False, error_msg

    try:
        keyring_username = _get_keyring_username(provider_lower)
        keyring.set_password(_KEYRING_SERVICE_NAME, keyring_username, api_key)
        return True, None
    except Exception as e:
        return False, f"Error saving API key to keyring: {e}"


def remove_api_key(provider: str) -> tuple[bool, str | None]:
    """
    Removes the API key for the specified provider from the system keyring.

    Note: This does not affect environment variables.

    Args:
        provider: The name of the provider (e.g., 'openai'). Case-insensitive.

    Returns:
        Tuple of (success, error_message)
        - success: True if the key was successfully removed or was not present
        - error_message: None if successful, error description if failed
    """
    provider_lower = provider.lower()
    provider_name = _PROVIDER_CONFIG.get(provider_lower, {}).get("name", provider.capitalize())
    keyring_username = _get_keyring_username(provider_lower)

    try:
        # Check if key exists in keyring before attempting delete
        current_key = keyring.get_password(_KEYRING_SERVICE_NAME, keyring_username)
        if current_key is None:
            return True, f"No {provider_name} key found in keyring."

        keyring.delete_password(_KEYRING_SERVICE_NAME, keyring_username)
        return True, f"{provider_name} API key successfully removed from keyring."
    except keyring.errors.PasswordDeleteError:
        # This specific error often means the password was not found
        return True, f"No {provider_name} key found in keyring to delete."
    except Exception as e:
        return False, f"Error removing {provider_name} API key from keyring: {e}"


def get_provider_display_name(provider: str) -> str:
    """
    Get the display name for a provider.

    Args:
        provider: The provider identifier (e.g., 'openai')

    Returns:
        The display name (e.g., 'OpenAI')
    """
    provider_lower = provider.lower()
    return _PROVIDER_CONFIG.get(provider_lower, {}).get("name", provider.capitalize())


def get_available_providers() -> list[str]:
    """
    Get list of all configured providers.

    Returns:
        List of provider identifiers
    """
    return list(_PROVIDER_CONFIG.keys())


def mask_api_key(api_key: str) -> str:
    """
    Create a masked version of an API key for display.

    Args:
        api_key: The API key to mask

    Returns:
        Masked API key showing only first and last few characters
    """
    if not api_key or len(api_key) < 8:
        return "****"

    mask_len = max(len(api_key) - 8, 0)
    return api_key[:4] + "*" * mask_len + api_key[-4:]
