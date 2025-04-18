"""
Chat interface UI components for Streamlit.
"""

from typing import Generator

import streamlit as st

from ..utils.formatting import (
    format_response_with_citations,
    format_retrieved_chunks_for_display,
    process_citations,
)
from .common import check_model_config_changes, render_page_header


def render_api_key_setup() -> None:
    """Render the API key setup screen when no keys are configured."""
    from paperchat.ui.api_settings import render_api_key_manager

    render_page_header("Welcome to PaperChat 🔑", "Configure your API key to get started")

    st.markdown("""
    To use PaperChat, you'll need at least one API key from a supported provider.
    Please configure an API key below to continue.
    """)

    tab1, tab2 = st.tabs(["OpenAI", "Anthropic"])

    with tab1:
        openai_configured = render_api_key_manager("openai", on_save_callback=lambda: st.rerun())
        st.markdown("""
        **Need an OpenAI API key?**
        1. Go to [OpenAI API Keys](https://platform.openai.com/api-keys)
        2. Create an account or sign in
        3. Click "Create new secret key"
        4. Copy the key and paste it above
        """)

    with tab2:
        anthropic_configured = render_api_key_manager(
            "anthropic", on_save_callback=lambda: st.rerun()
        )
        st.markdown("""
        **Need an Anthropic API key?**
        1. Go to [Anthropic Console](https://console.anthropic.com/)
        2. Create an account or sign in
        3. Navigate to API Keys and create a new key
        4. Copy the key and paste it above
        """)

    if (openai_configured or anthropic_configured) and st.button(
        "Continue to PaperChat", use_container_width=True
    ):
        st.session_state.show_api_setup = False
        st.rerun()


def _get_retrieval_filter() -> str | None:
    """Determine the Milvus filter expression based on selected documents."""
    selected_docs = st.session_state.get("selected_documents")
    filter_expression = None

    if not selected_docs:
        st.info("ℹ️ No specific documents selected. Searching across the entire knowledge base.")
        return None

    # build the filter basd on list items
    if isinstance(selected_docs, list) and len(selected_docs) > 0:
        # construct a Milvus 'in' filter expression
        # e.g., source in ["doc1", "doc2"]
        formatted_sources = [f'"{doc_name}"' for doc_name in selected_docs]
        filter_expression = f"source in [{', '.join(formatted_sources)}]"
    else:
        st.warning("Selected documents state is invalid. Searching across all documents.")
        # fallback to searching all if state is weird
        filter_expression = None

    return filter_expression


def _handle_rag_query(
    user_query: str, filter_expression: str | None, top_k: int
) -> tuple[Generator[str, None, None], list] | tuple[None, None]:
    """
    Execute RAG pipeline: retrieve chunks and return generator stream + retrieved chunks.

    Args:
        user_query: The user's query string.
        filter_expression: Milvus filter expression for retrieval.
        top_k: Number of chunks to retrieve.

    Returns:
        A tuple containing the LLM response generator and the list of retrieved chunks,
        or (None, None) if an error occurs.
    """
    try:
        rag_pipeline = st.session_state.rag_pipeline

        retrieved_chunks = rag_pipeline.retrieve(
            query=user_query, top_k=top_k, filter_expression=filter_expression
        )

        llm_response_stream = rag_pipeline.generate(
            query=user_query,
            retrieved_results=retrieved_chunks,
            stream=True,
        )
        return llm_response_stream, retrieved_chunks

    except Exception as e:
        error_message = f"Sorry, I encountered an error: {e!s}"
        st.error(error_message)
        return None, None


def render_chat_interface() -> None:
    """Render the chat interface."""
    chat_container = st.container()

    with chat_container:
        if st.session_state.messages:
            for msg in st.session_state.messages:
                with st.chat_message(msg["role"]):
                    if msg["role"] == "assistant":
                        formatted_content = format_response_with_citations(msg["content"])
                        st.markdown(formatted_content, unsafe_allow_html=True)
                        if msg.get("sources"):
                            with st.expander("📚 Sources", expanded=False):
                                sources_md = format_retrieved_chunks_for_display(msg["sources"])
                                st.markdown(sources_md, unsafe_allow_html=True)
                    else:
                        st.markdown(msg["content"])

    chat_input_placeholder = "Ask a question..."
    user_query = st.chat_input(chat_input_placeholder)

    if user_query:
        st.session_state.messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        with st.chat_message("assistant"):
            response_placeholder = st.empty()
            response_placeholder.markdown("Thinking...▌")

            check_model_config_changes()
            top_k = st.session_state.config.get("top_k", 5)

            filter_expression = _get_retrieval_filter()
            llm_response_stream, retrieved_chunks = _handle_rag_query(
                user_query, filter_expression, top_k
            )

            if llm_response_stream and retrieved_chunks is not None:
                full_response = response_placeholder.write_stream(llm_response_stream)

                processed_response, cited_chunks_filtered = process_citations(
                    full_response, retrieved_chunks
                )

                formatted_final_content = format_response_with_citations(processed_response)
                response_placeholder.markdown(formatted_final_content, unsafe_allow_html=True)

                message_data = {
                    "role": "assistant",
                    "content": processed_response,
                    "sources": cited_chunks_filtered,
                }
                st.session_state.messages.append(message_data)

                if cited_chunks_filtered:
                    with st.expander("📚 Sources", expanded=False):
                        sources_md = format_retrieved_chunks_for_display(cited_chunks_filtered)
                        st.markdown(sources_md, unsafe_allow_html=True)
            else:
                response_placeholder.error("An error occurred while processing your request.")
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": "Sorry, I couldn't process your request due to an error.",
                        "sources": [],
                    }
                )


def render_main_content() -> None:
    """Render the main content area with chat or welcome screen."""
    selected_docs = st.session_state.get("selected_documents")

    if selected_docs and isinstance(selected_docs, list) and len(selected_docs) > 0:
        render_chat_interface()
    elif selected_docs is not None and isinstance(selected_docs, list) and len(selected_docs) == 0:
        st.info(
            "Please select one or more documents from the 'Select Chat Context' dropdown in the sidebar to begin chatting."
        )
    else:
        # this branch covers cases where selected_documents hasn't been initialized yet
        # (e.g., on first run or if pipeline failed)
        st.markdown("""
        ### 👋 Welcome to PaperChat!

        To get started:
        1. Upload one or more PDFs using the sidebar
        2. Select the document(s) you want to chat with
        3. Ask questions about the document content

        PaperChat will use AI to retrieve relevant information and provide answers based on the selected document(s).
        """)
