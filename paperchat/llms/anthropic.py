"""
Anthropic provider implementation.
"""

from typing import Any, Generator

from anthropic import Anthropic

from ..utils.config import get_api_key
from ..utils.settings import DEFAULT_SYSTEM_PROMPT


def list_models() -> list[dict[str, Any]]:
    """List available models from Anthropic."""
    return [
        {"id": "claude-3-7-sonnet-20250219"},
        {"id": "claude-3-5-sonnet-20241022"},
        {"id": "claude-3-5-haiku-20241022"},
        {"id": "claude-3-opus-20240229"},
    ]


def get_client(api_key: str | None = None) -> Anthropic:
    """Get an Anthropic client with appropriate API key."""
    if not api_key:
        api_key = get_api_key("anthropic")
        if not api_key:
            return None
    return Anthropic(api_key=api_key)


def build_prompt(query: str, context: str) -> dict[str, Any]:
    """Build a prompt for the Anthropic API."""
    return {
        "system": DEFAULT_SYSTEM_PROMPT,
        "messages": [
            {
                "role": "user",
                "content": f"Here is the relevant context from the paper:\n\n{context}\n\nQuestion: {query}",
            }
        ],
    }


def generate_response(
    query: str,
    context: str,
    model: str,
    temperature: float,
    max_tokens: int,
    client: Anthropic | None = None,
) -> str:
    """Generate a response from Anthropic."""
    if client is None:
        client = get_client()
        if client is None:
            return "Error: Unable to create Anthropic client. Please check your API key."

    prompt = build_prompt(query, context)

    try:
        message = client.messages.create(
            model=model,
            system=prompt["system"],
            messages=prompt["messages"],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return message.content[0].text

    except Exception as e:
        return f"Error generating response: {e!s}"


def generate_streaming_response(
    query: str,
    context: str,
    model: str,
    temperature: float,
    max_tokens: int,
    client: Anthropic | None = None,
) -> Generator[str, None, None]:
    """Generate a streaming response from Anthropic."""
    if client is None:
        client = get_client()
    prompt = build_prompt(query, context)

    try:
        with client.messages.stream(
            model=model,
            system=prompt["system"],
            messages=prompt["messages"],
            temperature=temperature,
            max_tokens=max_tokens,
        ) as stream:
            for text in stream.text_stream:
                yield text

    except Exception as e:
        yield f"Error generating response: {e!s}"
