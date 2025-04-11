"""
Common UI components.
"""

import streamlit as st

from paperchat.core import RAGPipeline
from paperchat.llms import LLMManager
from paperchat.utils.api_keys import get_api_key
from paperchat.utils.config import DEFAULT_MODEL, DEFAULT_PROVIDER


def set_css_styles() -> None:
    """
    Apply custom CSS styles to improve the UI appearance.

    This function adds custom CSS to the Streamlit app to enhance the appearance
    of various UI elements.
    """
    st.markdown(
        """
        <style>
        /* Main content area styles */
        .main .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
        }

        /* Chat message styling */
        .stChatMessage {
            border-radius: 10px !important;
            margin-bottom: 0.8rem !important;
        }

        .stChatMessage p {
            margin-bottom: 0.8rem;
        }

        .stChatMessage p:last-child {
            margin-bottom: 0 !important;
        }

        /* Make first paragraph in chat messages look better */
        .stChatMessage div {
            line-height: 1.5;
        }

        /* Code blocks in chat */
        .stChatMessage code {
            font-size: 0.9rem !important;
            font-family: 'SFMono-Regular', Consolas, 'Liberation Mono', Menlo, Courier, monospace !important;
        }

        .stChatMessage pre {
            margin-top: 0.5rem !important;
            margin-bottom: 0.5rem !important;
            padding: 0.5rem !important;
            border-radius: 5px !important;
            background-color: #f0f0f0 !important;
        }

        /* Chat headers */
        .stChatMessage h1, .stChatMessage h2, .stChatMessage h3 {
            margin-top: 1rem !important;
            margin-bottom: 0.5rem !important;
        }

        /* Compact list styles in chat */
        .stChatMessage ul, .stChatMessage ol {
            margin-top: 0 !important;
            margin-bottom: 0.5rem !important;
            padding-left: 1.5rem !important;
        }

        /* Table styles for citations */
        .citations-table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 1rem;
            margin-bottom: 1rem;
            font-size: 0.9rem;
        }

        .citations-table th, .citations-table td {
            padding: 0.5rem;
            border: 1px solid #e0e0e0;
            text-align: left;
        }

        .citations-table th {
            background-color: #f6f6f6;
            font-weight: 600;
        }

        /* Nicer alerts */
        .alert {
            padding: 0.8rem;
            margin-bottom: 1rem;
            border-radius: 8px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def refresh_model_state() -> None:
    """
    Refresh LLM manager and RAG pipeline with current config settings.

    This function should be called whenever model settings change to ensure
    the LLM provider and model are correctly updated.
    """
    provider_name = st.session_state.config.get("provider_name", DEFAULT_PROVIDER)
    api_key = get_api_key(provider_name)

    st.session_state.llm_manager = LLMManager(config=st.session_state.config, api_key=api_key)
    st.session_state.rag_pipeline = RAGPipeline(
        llm_manager=st.session_state.llm_manager,
        embedding_model=st.session_state.embedding_model,
    )

    if "previous_config" not in st.session_state:
        st.session_state.previous_config = {}

    st.session_state.previous_config = {
        "provider_name": provider_name,
        "model_id": st.session_state.config.get("model_id", DEFAULT_MODEL),
        "temperature": st.session_state.config.get("temperature", 0.7),
    }


def check_model_config_changes() -> bool:
    """
    Check if model configuration has changed and needs refresh.

    Returns:
        bool: True if config changed and model state was refreshed
    """
    if "previous_config" not in st.session_state:
        refresh_model_state()
        return True

    prev_config = st.session_state.previous_config
    current_config = st.session_state.config
    # check for change in config
    if prev_config.get("provider_name") != current_config.get("provider_name") or prev_config.get(
        "model_id"
    ) != current_config.get("model_id"):
        refresh_model_state()
        return True

    return False


def render_page_header(title: str, subtitle: str | None = None) -> None:
    """
    Render a consistent page header with title and optional subtitle.

    Args:
        title: Main page title
        subtitle: Optional subtitle text
    """
    st.title(title)
    if subtitle:
        st.markdown(f"**{subtitle}**")
    st.divider()


def show_success(message: str) -> None:
    """
    Display a success message with consistent styling.

    Args:
        message: Success message to display
    """
    st.success(message)


def show_info(message: str) -> None:
    """
    Display an information message with consistent styling.

    Args:
        message: Info message to display
    """
    st.info(message)


def show_warning(message: str) -> None:
    """
    Display a warning message with consistent styling.

    Args:
        message: Warning message to display
    """
    st.warning(message)


def show_error(message: str) -> None:
    """
    Display an error message with consistent styling.

    Args:
        message: Error message to display
    """
    st.error(message)
