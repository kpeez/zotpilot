"""
Chat interface UI components for Streamlit.
"""

import streamlit as st

from paperchat.llm import rag_pipeline
from paperchat.ui.common import render_page_header, show_info
from paperchat.utils.formatting import (
    format_response_with_citations,
    format_retrieved_chunks_for_display,
    process_citations,
)


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

    # allow user to continue if at least one key is configured
    if (openai_configured or anthropic_configured) and st.button(
        "Continue to PaperChat", use_container_width=True
    ):
        st.session_state.show_api_setup = False
        st.rerun()


def render_chat_interface() -> None:
    """Render the chat interface."""
    chat_container = st.container()

    with chat_container:
        if st.session_state.messages:
            for msg in st.session_state.messages:
                with st.chat_message(msg["role"]):
                    if msg["role"] == "assistant":
                        st.markdown(msg["content"], unsafe_allow_html=True)
                        if msg.get("sources"):
                            with st.expander("📚 Sources", expanded=False):
                                sources_md = format_retrieved_chunks_for_display(msg["sources"])
                                st.markdown(sources_md, unsafe_allow_html=True)
                    else:
                        st.markdown(msg["content"])

    user_query = st.chat_input(
        f"Ask a question about {st.session_state.filenames[st.session_state.active_document_hash]}"
    )

    if user_query:
        st.session_state.messages.append({"role": "user", "content": user_query})

        with st.chat_message("user"):
            st.markdown(user_query)

        response_placeholder = st.empty()

        with st.spinner("Generating response..."):
            try:
                active_doc_data = st.session_state.processed_documents[
                    st.session_state.active_document_hash
                ]
                settings = st.session_state.settings

                llm_response, retrieved_chunks = rag_pipeline(
                    query=user_query,
                    document_data=active_doc_data,
                    top_k=settings["top_k"],
                    model=settings["model"],
                    temperature=settings["temperature"],
                    max_tokens=settings["max_tokens"],
                    stream=False,
                    embedding_model=st.session_state.embedding_model,
                    client=st.session_state.llm_client,
                )

                processed_response, cited_chunks_filtered = process_citations(
                    llm_response, retrieved_chunks
                )

                formatted_response_html = format_response_with_citations(processed_response)

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": formatted_response_html,
                        "raw_content": llm_response,
                        "sources": cited_chunks_filtered,
                    }
                )

                with response_placeholder.chat_message("assistant"):
                    st.markdown(formatted_response_html, unsafe_allow_html=True)

                    if cited_chunks_filtered:
                        with st.expander("📚 Sources", expanded=False):
                            sources_md = format_retrieved_chunks_for_display(cited_chunks_filtered)
                            st.markdown(sources_md, unsafe_allow_html=True)

            except Exception as e:
                error_message = f"Error: {e!s}"
                st.error(error_message)

                st.session_state.messages.append(
                    {"role": "assistant", "content": error_message, "sources": []}
                )

                with response_placeholder.chat_message("assistant"):
                    st.error(error_message)


def render_main_content() -> None:
    """Render the main content area with chat or welcome screen."""
    if not st.session_state.processed_documents:
        st.markdown("""
        ### 👋 Welcome to PaperChat!

        To get started:
        1. Upload one or more PDFs using the sidebar
        2. Select the document you want to chat with
        3. Ask questions about the document content

        PaperChat will use AI to retrieve relevant information and provide answers based on the selected document.
        """)
    elif st.session_state.active_document_hash is None and st.session_state.processed_documents:
        show_info("Please select a document from the sidebar to start chatting.")
    else:
        render_chat_interface()
