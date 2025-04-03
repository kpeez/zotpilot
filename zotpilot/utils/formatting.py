import re
from typing import Any


def format_context(results: list[dict[str, Any]], include_metadata: bool = True) -> str:
    """
    Format retrieved chunks into a context string for the LLM.

    Args:
        results: List of retrieval results
        include_metadata: Whether to include metadata in the context

    Returns:
        Formatted context string
    """
    context_parts = []

    for i, result in enumerate(results):
        if include_metadata:
            context_parts.append(
                f"[CHUNK {i + 1} (Page {result['metadata']['page']})]:\n{result['text']}\n"
            )
        else:
            context_parts.append(result["text"])

    return "\n\n".join(context_parts)


def format_response_with_citations(response: str) -> str:
    """
    Format the response with highlighted citations.

    Args:
        response: Response text from the LLM

    Returns:
        Response with HTML-formatted citation highlights
    """
    citation_pattern = r"\[(\d+(?:,\s*\d+)*)\]"
    formatted_response = re.sub(
        citation_pattern,
        lambda m: f'<span style="background-color: #e6f3ff; padding: 1px 4px; border-radius: 3px;">{m.group(0)}</span>',
        response,
    )
    return formatted_response


def format_retrieved_chunks_for_display(chunks: list[dict[str, Any]]) -> str:
    """
    Format retrieved chunks for display in the UI.

    Args:
        chunks: List of retrieved chunk data

    Returns:
        Markdown-formatted text for displaying chunks in Streamlit
    """
    if not chunks:
        return "No sources retrieved."

    formatted_text = "### Sources\n\n"
    for i, chunk in enumerate(chunks):
        page_num = chunk.get("metadata", {}).get("page", "Unknown page")
        text = chunk.get("text", "")

        max_display_length = 300
        if len(text) > max_display_length:
            text = text[:max_display_length] + "..."

        formatted_text += f"**Source [{i + 1}]** (Page {page_num})\n\n"
        formatted_text += f"{text}\n\n---\n\n"

    return formatted_text
