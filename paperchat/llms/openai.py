"""
OpenAI provider implementation.
"""

from typing import Any, Generator

from openai import OpenAI

from ..utils.config import get_api_key
from ..utils.settings import DEFAULT_SYSTEM_PROMPT


def list_models() -> list[dict[str, Any]]:
    """List available models from OpenAI."""
    return [
        {"id": "gpt-4o"},
        {"id": "gpt-4o-mini"},
    ]


def get_client(api_key: str | None = None) -> OpenAI:
    """Get an OpenAI client with appropriate API key."""
    if not api_key:
        api_key = get_api_key("openai")
        if not api_key:
            return None

    return OpenAI(api_key=api_key)


def build_prompt(query: str, context: str) -> list[dict[str, str]]:
    """Build a prompt for the OpenAI API."""
    return [
        {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"Here is the relevant context from the paper:\n\n{context}\n\nQuestion: {query}",
        },
    ]


def generate_response(
    query: str,
    context: str,
    model: str,
    temperature: float,
    max_tokens: int,
    client: OpenAI | None = None,
) -> str:
    """Generate a response from OpenAI."""
    if client is None:
        client = get_client()

    messages = build_prompt(query, context)
    try:
        completion = client.chat.completions.create(
            model=model, messages=messages, temperature=temperature, max_tokens=max_tokens
        )
        return completion.choices[0].message.content

    except Exception as e:
        return f"Error generating response: {e!s}"


def generate_streaming_response(
    query: str,
    context: str,
    model: str,
    temperature: float,
    max_tokens: int,
    client: OpenAI | None = None,
) -> Generator[str, None, None]:
    """Generate a streaming response from OpenAI."""
    if client is None:
        client = get_client()

    messages = build_prompt(query, context)
    try:
        stream = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
        )

        for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    except Exception as e:
        yield f"Error generating response: {e!s}"
