from typing import Any, Generator

from .common import get_provider


class LLMManager:
    """Manager class for handling LLM providers and clients."""

    def __init__(self, config: dict[str, Any], api_key: str | None = None):
        """Initialize the LLM manager with a configuration dictionary.

        Args:
            config: Dictionary with 'provider_name', 'model_id', 'temperature', 'max_tokens'
            api_key: Optional API key for the provider
        """
        self.config = config
        self._api_key = api_key
        self._client = None

    @property
    def client(self) -> Any:
        """Get or create the LLM client."""
        if self._client is None:
            self._client = self._create_client()
        return self._client

    def _create_client(self) -> Any:
        """Create a new client for the configured provider."""
        provider = get_provider(self.config["provider_name"])
        return provider.get_client(self._api_key)

    def generate_response(self, query: str, context: str) -> str:
        """Generate a response from the LLM."""
        provider = get_provider(self.config["provider_name"])
        return provider.generate_response(
            query=query,
            context=context,
            model=self.config["model_id"],
            temperature=self.config["temperature"],
            max_tokens=self.config["max_tokens"],
            client=self.client,
        )

    def generate_streaming_response(self, query: str, context: str) -> Generator[str, None, None]:
        """Generate a streaming response from the LLM."""
        provider = get_provider(self.config["provider_name"])
        yield from provider.generate_streaming_response(
            query=query,
            context=context,
            model=self.config["model_id"],
            temperature=self.config["temperature"],
            max_tokens=self.config["max_tokens"],
            client=self.client,
        )

    def update_config(self, **kwargs) -> None:
        """Update the configuration with new values."""
        provider_changed = "provider_name" in kwargs

        self.config.update(kwargs)

        if provider_changed:
            self._client = None
