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


def process_citations(
    raw_response: str, retrieved_chunks: list[dict[str, Any]]
) -> tuple[str, list[dict[str, Any]]]:
    """
    Processes citations in the raw LLM response to use sequential footnote-style
    references and filters chunks to only include cited ones.

    Finds citations like (Chunk 1, Page 5) or (Source 3, Page 10), replaces them
    with sequential markers like [1], [2], etc., based on the order of appearance,
    and returns the processed response and the list of corresponding cited chunks
    in the correct order.

    Args:
        raw_response: The raw text response from the LLM.
        retrieved_chunks: The list of all retrieved chunk dictionaries.

    Returns:
        A tuple containing:
            - The response string with sequential citations ([1], [2], ...).
            - A list of chunk dictionaries that were actually cited, ordered
              by their citation number.
    """
    # citation pattern: (Chunk 1, Page 5) or (Source 3, Page 10)
    # group 1 is chunk/source number, group 2 is page number
    citation_pattern = re.compile(r"\((?:Chunk|Source)\s*(\d+)[,\s]*Page\s*(\d+)\)")

    cited_chunk_indices: list[int] = []
    # map original chunk index -> citation number [1], [2]...
    unique_citations: dict[int, int] = {}
    # map full citation string -> footnote string e.g. "(Chunk 1, Page 5)" -> "[1]"
    citation_map: dict[str, str] = {}

    def replace_citation(match: re.Match) -> str:
        # get the full matched string, e.g., "(Chunk 1, Page 5)"
        full_match, chunk_num_str, _page_num_str = match.group(0), match.group(1), match.group(2)

        try:
            # convert Docling's 1-based chunk index to 0-based for list access
            original_chunk_index = int(chunk_num_str) - 1
        except ValueError:
            return full_match

        if not (0 <= original_chunk_index < len(retrieved_chunks)):
            return full_match

        if full_match not in citation_map:
            if original_chunk_index not in unique_citations:
                current_citation_num = len(unique_citations) + 1
                unique_citations[original_chunk_index] = current_citation_num
                cited_chunk_indices.append(original_chunk_index)
                citation_map[full_match] = f"[{current_citation_num}]"
            else:
                # already seen this chunk index, map the full string variation to existing number
                citation_map[full_match] = f"[{unique_citations[original_chunk_index]}]"

        return citation_map[full_match]

    processed_response = citation_pattern.sub(replace_citation, raw_response)

    # filter and order the chunks based on the cited_chunk_indices
    cited_chunks_ordered = [
        retrieved_chunks[i] for i in cited_chunk_indices if 0 <= i < len(retrieved_chunks)
    ]

    return processed_response, cited_chunks_ordered


def format_response_with_citations(response: str) -> str:
    """
    Format the response with highlighted citations. Now expects [1], [2] style.

    Args:
        response: Response text from the LLM, potentially with [N] citations.

    Returns:
        Response with HTML-formatted citation highlights
    """
    # find [1], [2] style citations
    citation_pattern = r"(\[\d+\])"
    formatted_response = re.sub(
        citation_pattern,
        lambda m: f'<span style="background-color: #e6f3ff; padding: 1px 4px; border-radius: 3px;">{m.group(1)}</span>',
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

        formatted_text += f"**Source [{i + 1}]** (Page {page_num})\n\n"
        formatted_text += f"{text}\n\n---\n\n"

    return formatted_text
