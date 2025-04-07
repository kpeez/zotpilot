import os

import streamlit as st

from paperchat.embeddings import EmbeddingModel
from paperchat.llm import get_openai_client
from paperchat.ui import (
    render_api_key_setup,
    render_main_content,
    render_settings_page,
    render_sidebar,
    set_css_styles,
)
from paperchat.utils.config import get_api_key

os.environ["TOKENIZERS_PARALLELISM"] = "false"

st.set_page_config(
    page_title="PaperChat - PDF Chat",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

set_css_styles()


def initialize_session():
    """Initialize all session state variables and models in one place."""
    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "processed_documents" not in st.session_state:
        # stores hash -> {processed data dict from process_document()}
        st.session_state.processed_documents = {}
    if "filenames" not in st.session_state:
        # stores hash -> original filename string
        st.session_state.filenames = {}
    if "active_document_hash" not in st.session_state:
        # hash of the currently selected doc for chatting
        st.session_state.active_document_hash = None

    # remove single-document state variables if they exist from previous runs
    if "document_data" in st.session_state:
        del st.session_state["document_data"]
    if "last_processed_hash" in st.session_state:
        del st.session_state["last_processed_hash"]

    if "settings" not in st.session_state:
        st.session_state.settings = {
            "top_k": 5,
            "temperature": 0.7,
            "max_tokens": 1000,
            "model": "gpt-4o",
        }

    if "embedding_model" not in st.session_state:
        st.session_state.embedding_model = EmbeddingModel()

    if "llm_client" not in st.session_state:
        st.session_state.llm_client = get_openai_client()

    # UI state management
    if "show_api_setup" not in st.session_state:
        st.session_state.show_api_setup = False
    if "show_settings" not in st.session_state:
        st.session_state.show_settings = False


def main():
    """Main entry point for the Streamlit app."""
    initialize_session()

    # check for API keys
    openai_key = get_api_key("openai")
    if not openai_key or st.session_state.show_api_setup:
        render_api_key_setup()
        st.stop()

    st.title("🤖 PaperChat - Chat with your research library")
    # show settings page if requested
    if st.session_state.show_settings:
        render_settings_page(on_save_callback=lambda: st.rerun())
        if st.button("⬅️ Back to Chat"):
            st.session_state.show_settings = False
            st.rerun()
        st.stop()

    # standard app UI (sidebar + main content)
    with st.sidebar:
        render_sidebar()

    render_main_content()


if __name__ == "__main__":
    main()
