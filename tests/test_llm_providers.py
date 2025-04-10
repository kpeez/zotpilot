from unittest import mock

from paperchat.llms import PROVIDERS, AnthropicAdapter, OpenAIAdapter, get_client, get_provider
from paperchat.utils.api_keys import get_api_key
from paperchat.utils.config import DEFAULT_PROVIDER


class TestLLMProviders:
    """Tests for the LLM provider functionality."""

    def test_get_provider(self):
        """Test retrieving provider classes."""
        default_provider = get_provider()
        assert default_provider == PROVIDERS[DEFAULT_PROVIDER]

        openai_provider = get_provider("openai")
        assert openai_provider == PROVIDERS["openai"]

        anthropic_provider = get_provider("anthropic")
        assert anthropic_provider == PROVIDERS["anthropic"]

    @mock.patch("paperchat.llms.openai.get_api_key")
    @mock.patch("paperchat.llms.anthropic.get_api_key")
    def test_get_client_direct_classes(self, mock_anthropic_get_key, mock_openai_get_key):
        """Test client creation by mocking at the class level."""
        mock_openai_get_key.return_value = "test-openai-key"
        mock_anthropic_get_key.return_value = "test-anthropic-key"

        openai_client = OpenAIAdapter.get_client()
        assert openai_client is not None
        mock_openai_get_key.assert_called_with("openai")

        anthropic_client = AnthropicAdapter.get_client()
        assert anthropic_client is not None
        mock_anthropic_get_key.assert_called_with("anthropic")

    @mock.patch("paperchat.llms.openai.get_api_key")
    @mock.patch("paperchat.llms.anthropic.get_api_key")
    def test_failed_client_creation(self, mock_anthropic_get_key, mock_openai_get_key):
        """Test how adapters handle missing API keys."""
        mock_openai_get_key.return_value = None
        mock_anthropic_get_key.return_value = None

        openai_client = OpenAIAdapter.get_client()
        assert openai_client is None

        anthropic_client = AnthropicAdapter.get_client()
        assert anthropic_client is None

    @mock.patch("paperchat.llms.openai.OpenAIAdapter.get_client")
    @mock.patch("paperchat.llms.anthropic.AnthropicAdapter.get_client")
    def test_common_client_creator(self, mock_anthropic_client, mock_openai_client):
        """Test the common client creation function with mocks."""
        mock_openai_client.return_value = "openai-client"
        mock_anthropic_client.return_value = "anthropic-client"

        client = get_client(provider_name="openai")
        assert client == "openai-client"
        mock_openai_client.assert_called_once()

        client = get_client(provider_name="anthropic")
        assert client == "anthropic-client"
        mock_anthropic_client.assert_called_once()

    def test_real_api_keys_available(self):
        """Test if real API keys are available in the environment."""
        openai_key = get_api_key("openai")
        anthropic_key = get_api_key("anthropic")

        print("\nAPI keys available:")
        print(f"OpenAI key available: {openai_key is not None}")
        print(f"Anthropic key available: {anthropic_key is not None}")

        assert openai_key is not None
        assert anthropic_key is not None
