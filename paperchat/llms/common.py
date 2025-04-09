"""
Common protocols and interfaces for LLM providers.
"""

from typing import Any, Generator, Protocol, Type, runtime_checkable

from ..utils.config import DEFAULT_PROVIDER
from .anthropic import AnthropicAdapter
from .openai import OpenAIAdapter


@runtime_checkable
class LLMProvider(Protocol):
    """Protocol defining the interface for LLM providers."""

    @staticmethod
    def get_client(api_key: str | None = None) -> Any:
        """Get a client for this provider."""
        ...

    @staticmethod
    def build_prompt(query: str, context: str) -> Any:
        """Build a provider-specific prompt."""
        ...

    @staticmethod
    def generate_response(
        query: str,
        context: str,
        model: str,
        temperature: float,
        max_tokens: int,
        client: Any | None = None,
    ) -> str:
        """Generate a response from the LLM."""
        ...

    @staticmethod
    def generate_streaming_response(
        query: str,
        context: str,
        model: str,
        temperature: float,
        max_tokens: int,
        client: Any | None = None,
    ) -> Generator[str, None, None]:
        """Generate a streaming response from the LLM."""
        ...

    @staticmethod
    def list_models(client: Any | None = None) -> list[dict[str, Any]]:
        """List available models from the provider."""
        ...


# registry of available LLM providers
PROVIDERS: dict[str, Type[LLMProvider]] = {
    "openai": OpenAIAdapter,
    "anthropic": AnthropicAdapter,
}


def get_provider(provider_name: str = DEFAULT_PROVIDER) -> Type[LLMProvider]:
    """
    Get the provider class for the specified provider name.

    Args:
        provider_name: Name of the provider to use

    Returns:
        Provider class

    Raises:
        ValueError: If the provider is not supported
    """
    provider = PROVIDERS.get(provider_name)
    if provider is None:
        supported = ", ".join(PROVIDERS.keys())
        raise ValueError(f"Unsupported provider: {provider_name}. Supported providers: {supported}")
    return provider


def get_client(provider_name: str = DEFAULT_PROVIDER, api_key: str | None = None) -> Any:
    """
    Get a client for the specified provider.

    Args:
        provider_name: Name of the provider to use
        api_key: Optional API key

    Returns:
        Provider client instance
    """
    provider = get_provider(provider_name)
    return provider.get_client(api_key)


def generate_response(
    query: str,
    context: str,
    model: str,
    temperature: float,
    max_tokens: int,
    client: Any | None = None,
    provider_name: str = DEFAULT_PROVIDER,
) -> str:
    """
    Generate a response using the specified provider.

    Args:
        query: User's question
        context: Context from retrieved document chunks
        model: Model to use
        temperature: Temperature parameter (0-1)
        max_tokens: Maximum tokens in response
        client: Optional client instance
        provider_name: Name of the provider to use

    Returns:
        Generated response
    """
    provider = get_provider(provider_name)
    return provider.generate_response(
        query=query,
        context=context,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        client=client,
    )


def generate_streaming_response(
    query: str,
    context: str,
    model: str,
    temperature: float,
    max_tokens: int,
    client: Any | None = None,
    provider_name: str = DEFAULT_PROVIDER,
) -> Generator[str, None, None]:
    """
    Generate a streaming response using the specified provider.

    Args:
        query: User's question
        context: Context from retrieved document chunks
        model: Model to use
        temperature: Temperature parameter (0-1)
        max_tokens: Maximum tokens in response
        client: Optional client instance
        provider_name: Name of the provider to use

    Yields:
        Chunks of the generated response
    """
    provider = get_provider(provider_name)
    yield from provider.generate_streaming_response(
        query=query,
        context=context,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        client=client,
    )


def register_provider(name: str, provider_class: Type[LLMProvider]) -> None:
    """
    Register a new LLM provider.

    Args:
        name: Name to register the provider under
        provider_class: Provider class implementing the LLMProvider protocol
    """
    PROVIDERS[name] = provider_class


def list_available_providers() -> list[str]:
    """
    List all available provider names.

    Returns:
        List of provider names
    """
    return list(PROVIDERS.keys())


def list_models(provider_name: str = DEFAULT_PROVIDER) -> list[dict[str, Any]]:
    """
    List models available from the specified provider.

    Args:
        provider_name: Name of the provider

    Returns:
        List of model information dictionaries
    """
    provider = get_provider(provider_name)
    return provider.list_models()
