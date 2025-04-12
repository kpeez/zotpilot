import os

import streamlit as st

from paperchat.core import EmbeddingModel, VectorStore
from paperchat.ui import (
    check_model_config_changes,
    refresh_model_state,
    render_api_key_setup,
    render_main_content,
    render_settings_page,
    render_sidebar,
    set_css_styles,
)
from paperchat.utils.api_keys import get_api_key
from paperchat.utils.config import DEFAULT_MAX_TOKENS, DEFAULT_MODEL, DEFAULT_PROVIDER

os.environ["TOKENIZERS_PARALLELISM"] = "false"

st.set_page_config(
    page_title="PaperChat - PDF Chat",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

set_css_styles()


def initialize_document_state() -> None:
    """Initialize document-related session state variables."""
    if "processed_documents" not in st.session_state:
        # stores hash -> {processed data dict from process_document()}
        st.session_state.processed_documents = {}
    if "filenames" not in st.session_state:
        # stores hash -> original filename string
        st.session_state.filenames = {}
    if "active_document_hash" not in st.session_state:
        # hash of the currently selected doc for chatting
        st.session_state.active_document_hash = None
    if "document_sources" not in st.session_state:
        # stores hash -> source info ("vector_store" or "session")
        st.session_state.document_sources = {}

    if "document_data" in st.session_state:
        del st.session_state["document_data"]
    if "last_processed_hash" in st.session_state:
        del st.session_state["last_processed_hash"]


def initialize_model_state() -> None:
    """Initialize model and settings-related session state variables."""
    if "config" not in st.session_state:
        st.session_state.config = {
            "provider_name": DEFAULT_PROVIDER,
            "model_id": DEFAULT_MODEL,
            "temperature": 0.7,
            "max_tokens": DEFAULT_MAX_TOKENS,
            "top_k": 5,
        }

    if "provider_settings" not in st.session_state:
        provider_name = st.session_state.config.get("provider_name", DEFAULT_PROVIDER)
        st.session_state.provider_settings = {
            provider_name: {
                "model": st.session_state.config.get("model_id", DEFAULT_MODEL),
                "temperature": st.session_state.config.get("temperature", 0.7),
            }
        }

    if "settings" not in st.session_state:
        st.session_state.settings = {"top_k": st.session_state.config.get("top_k", 5)}

    if "embedding_model" not in st.session_state:
        st.session_state.embedding_model = EmbeddingModel()

    if "vector_store" not in st.session_state:
        st.session_state.vector_store = VectorStore()

    if "llm_manager" not in st.session_state or "rag_pipeline" not in st.session_state:
        refresh_model_state()


def initialize_session() -> None:
    """Initialize all session state variables and components in one place."""
    if "messages" not in st.session_state:
        st.session_state.messages = []

    initialize_document_state()
    initialize_model_state()

    if "show_api_setup" not in st.session_state:
        st.session_state.show_api_setup = False
    if "show_settings" not in st.session_state:
        st.session_state.show_settings = False


def main() -> None:
    """Main entry point for the Streamlit app."""
    initialize_session()
    check_model_config_changes()
    provider_name = st.session_state.config.get("provider_name", DEFAULT_PROVIDER)
    provider_key = get_api_key(provider_name)

    if not provider_key or st.session_state.show_api_setup:
        render_api_key_setup()
        st.stop()

    st.title("🤖 PaperChat - Chat with your research library")
    if st.session_state.show_settings:
        render_settings_page(on_save_callback=lambda: refresh_model_state())
        st.stop()

    with st.sidebar:
        render_sidebar()

    render_main_content()


if __name__ == "__main__":
    main()
