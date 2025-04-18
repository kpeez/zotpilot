import re
from typing import Any


def format_context(retrieved_results: list[dict[str, Any]], include_metadata: bool = True) -> str:
    """
    Format retrieved chunks into a context string for the LLM.

    Args:
        retrieved_results: List of retrieval results
        include_metadata: Whether to include metadata in the context

    Returns:
        Formatted context string
    """
    context_parts = []

    for i, result in enumerate(retrieved_results):
        if include_metadata:
            text = result["entity"]["text"]
            pages = result["entity"].get("page_numbers", [])
            page_str = ", ".join(map(str, pages)) if pages else "Unknown"
            source = result["entity"]["source"]
            context_parts.append(f"[CHUNK {i + 1} | Source: {source} | {page_str}]:\n{text}\n")
        else:
            context_parts.append(f"[CHUNK {i + 1}]:\n{text}\n")

    return "\n\n".join(context_parts)


def process_citations(
    raw_response: str, retrieved_chunks: list[dict[str, Any]]
) -> tuple[str, list[dict[str, Any]]]:
    """
    Processes citations in the raw LLM response to use sequential footnote-style
    references and filters chunks to only include cited ones.

    Finds citations like "Chunk 1", "(see CHUNK 1)", "(Chunks 1, 2)", etc., replaces them
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
    # captures the chunk type (group 1) and the number list (group 2).
    citation_pattern = re.compile(r"\b(Chunks?)\s+(\d+(?:\s*(?:,|and)\s*\d+)*)\b", re.IGNORECASE)

    cited_chunk_indices: list[int] = []
    unique_citations: dict[int, int] = {}  # map original chunk index -> citation number [1], [2]...
    processed_parts: list[str] = []
    last_end = 0

    for match in citation_pattern.finditer(raw_response):
        start, end = match.span()
        processed_parts.append(raw_response[last_end:start])
        chunk_nums_str = match.group(2)
        chunk_indices_0_based = []
        try:
            # use more robust split logic allowing flexible spacing
            indices_1_based = [int(n.strip()) for n in re.split(r"\s*(?:and|,)\s*", chunk_nums_str)]
            chunk_indices_0_based = [idx - 1 for idx in indices_1_based]
        except ValueError:
            # if number parsing fails, append the original matched text and skip
            processed_parts.append(match.group(0))
            last_end = end
            continue

        # validate chunk indices
        if not all(0 <= idx < len(retrieved_chunks) for idx in chunk_indices_0_based):
            processed_parts.append(match.group(0))  # append original if invalid index
            last_end = end
            continue

        # generate citation numbers and update tracking
        citation_numbers_for_this_match = []
        for original_chunk_index in chunk_indices_0_based:
            if original_chunk_index not in unique_citations:
                current_citation_num = len(unique_citations) + 1
                unique_citations[original_chunk_index] = current_citation_num
                cited_chunk_indices.append(original_chunk_index)
                citation_numbers_for_this_match.append(str(current_citation_num))
            else:
                citation_numbers_for_this_match.append(str(unique_citations[original_chunk_index]))

        citation_numbers_for_this_match.sort(key=int)
        # create the replacement footnote string, e.g., "[1]" or "[1, 2]"
        replacement_string = f"[{', '.join(citation_numbers_for_this_match)}]"
        processed_parts.append(replacement_string)

        last_end = end

    # append any remaining text after the last match
    processed_parts.append(raw_response[last_end:])

    processed_response = "".join(processed_parts)

    # ensure cited chunks are ordered by their *first* appearance (citation number)
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
    # find [1] or [1, 2] style citations
    citation_pattern = r"(\[\d+(?:,\s*\d+)*\])"
    formatted_response = re.sub(
        citation_pattern,
        lambda m: f'<span class="citation-marker">{m.group(1)}</span>',
        response,
    )
    return formatted_response


def format_retrieved_chunks_for_display(chunks: list[dict[str, Any]]) -> str:
    """
    Format retrieved chunks for display in the UI.
    Handles the Milvus result structure.

    Args:
        chunks: List of retrieved chunk dictionaries from Milvus search.
                Expected structure: [{'id': ..., 'distance': ..., 'entity': {...}}]

    Returns:
        Markdown-formatted text for displaying chunks in Streamlit
    """
    if not chunks:
        return "No sources retrieved."

    formatted_text = "### Sources\n\n"
    for i, result in enumerate(chunks):
        entity = result.get("entity", {})
        text = entity.get("text", "No text found.")
        page_numbers = entity.get("page_numbers", [])
        source = entity.get("source", "Unknown source")

        if page_numbers:
            page_str = ", ".join(map(str, sorted(set(page_numbers))))
            page_info = f"Page(s) {page_str}"
        else:
            page_info = "Unknown page"

        formatted_text += f"**Source [{i + 1}]** ({source} | {page_info})\n\n"
        formatted_text += f"{text}\n\n---\n\n"

    return formatted_text
