from typing import Any, Generator, Type

from ..llm import LLMConfig
from .common import LLMProvider, get_provider


class LLMManager:
    """Manager class for handling LLM providers and clients."""

    def __init__(self, config: LLMConfig | None = None, api_key: str | None = None):
        """Initialize the LLM manager with a configuration."""
        self.config = config or LLMConfig()
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
        provider = self._get_provider()
        return provider.get_client(self._api_key)

    def _get_provider(self) -> Type[LLMProvider]:
        """Get the provider class for the configured provider name."""
        return get_provider(self.config.provider_name)

    def generate_response(self, query: str, context: str) -> str:
        """Generate a response from the LLM."""
        provider = self._get_provider()
        return provider.generate_response(
            query=query,
            context=context,
            model=self.config.model_id,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            client=self.client,
        )

    def generate_streaming_response(self, query: str, context: str) -> Generator[str, None, None]:
        """Generate a streaming response from the LLM."""
        provider = self._get_provider()
        yield from provider.generate_streaming_response(
            query=query,
            context=context,
            model=self.config.model_id,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            client=self.client,
        )

    def update_config(self, **kwargs) -> None:
        """Update the configuration with new values."""
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)

        # reset client if the provider changes
        if "provider_name" in kwargs:
            self._client = None
