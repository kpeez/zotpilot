"""
Gemini provider implementation.
"""

from typing import Any, Generator

from google import genai
from google.genai import types

from ..utils.api_keys import get_api_key
from ..utils.config import DEFAULT_SYSTEM_PROMPT


class GeminiAdapter:
    """Gemini provider implementation."""

    @staticmethod
    def list_models(client: Any | None = None) -> list[dict[str, Any]]:
        """List available models from Gemini."""
        return [
            {"id": "gemini-2.5-pro-preview-03-25"},
            {"id": "gemini-2.0-flash"},
        ]

    @staticmethod
    def get_client(api_key: str | None = None) -> Any:
        """Get a Gemini client with appropriate API key."""
        if not api_key:
            api_key = get_api_key("gemini")
            if not api_key:
                return None

        return genai.Client(api_key=api_key)

    @staticmethod
    def build_prompt(query: str, context: str) -> str:
        """Build a prompt for the Gemini API."""
        return f"Here is the relevant context from the paper:\n\n{context}\n\nQuestion: {query}"

    @staticmethod
    def generate_response(
        query: str,
        context: str,
        model: str,
        temperature: float,
        max_tokens: int,
        client: Any | None = None,
    ) -> str:
        """Generate a response from Gemini."""
        if client is None:
            client = GeminiAdapter.get_client()
            if client is None:
                return "Error: Unable to create Gemini client. Please check your API key."

        prompt = GeminiAdapter.build_prompt(query, context)

        try:
            response = client.models.generate_content(
                model=model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=temperature,
                    max_output_tokens=max_tokens,
                    system_instruction=DEFAULT_SYSTEM_PROMPT,
                ),
            )
            return response.text

        except Exception as e:
            return f"Error generating response: {e!s}"

    @staticmethod
    def generate_streaming_response(
        query: str,
        context: str,
        model: str,
        temperature: float,
        max_tokens: int,
        client: Any | None = None,
    ) -> Generator[str, None, None]:
        """Generate a streaming response from Gemini."""
        if client is None:
            client = GeminiAdapter.get_client()
            if client is None:
                yield "Error: Unable to create Gemini client. Please check your API key."
                return

        prompt = GeminiAdapter.build_prompt(query, context)

        try:
            for chunk in client.models.generate_content_stream(
                model=model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=temperature,
                    max_output_tokens=max_tokens,
                    system_instruction=DEFAULT_SYSTEM_PROMPT,
                ),
            ):
                if chunk.text:
                    yield chunk.text

        except Exception as e:
            yield f"Error generating response: {e!s}"
