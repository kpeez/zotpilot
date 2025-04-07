"""
Settings UI components for Streamlit.
"""

from typing import Callable

import streamlit as st

from paperchat.ui.api_settings import render_api_key_manager
from paperchat.ui.common import render_page_header, show_info
from paperchat.utils.config import get_api_key, get_available_providers

PROVIDER_MODELS = {
    "openai": ["gpt-4o", "gpt-4o-mini"],
    "anthropic": ["claude-3.7-sonnet", "claude-3.5-sonnet", "claude-3.5-haiku"],
    # TODO: add gemini models
}


def render_model_settings(provider: str) -> None:
    """
    Render model-specific settings UI for a provider.

    Args:
        provider: Provider name (e.g., "openai")
    """
    provider = provider.lower()
    models = PROVIDER_MODELS.get(provider, [])
    if models:
        st.selectbox(
            "Default model",
            models,
            help="Select which model to use by default",
            key=f"{provider}_default_model",
        )

    # temperature setting (common across providers)
    st.slider(
        "Temperature",
        min_value=0.0,
        max_value=2.0,
        value=0.7,
        step=0.1,
        help="Controls randomness in responses: 0=deterministic, 1=creative",
        key=f"{provider}_temperature",
    )


def render_provider_settings(provider: str, on_save_callback: Callable | None = None):
    """
    Render settings for a specific provider, including API key and model settings.

    Args:
        provider: Provider name (e.g., "openai")
        on_save_callback: Function to call after saving settings
    """
    api_key_configured = render_api_key_manager(provider, on_save_callback)
    if api_key_configured:
        st.divider()
        st.subheader("Model Settings")
        render_model_settings(provider)

    st.divider()
    if provider == "openai":
        st.markdown("[Get an OpenAI API key](https://platform.openai.com/api-keys)")
    elif provider == "anthropic":
        st.markdown("[Get an Anthropic API key](https://console.anthropic.com/)")


def render_settings_tabs(on_save_callback: Callable | None = None):
    """
    Render settings as tabs, one for each provider.

    Args:
        on_save_callback: Function to call after saving settings
    """
    providers = get_available_providers()

    if not providers:
        show_info("No API providers configured.")
        return

    tab_labels = [provider.capitalize() for provider in providers]
    tabs = st.tabs(tab_labels)

    for i, provider in enumerate(providers):
        with tabs[i]:
            render_provider_settings(provider, on_save_callback)


def render_settings_page(on_save_callback: Callable | None = None):
    """
    Render a complete settings page.

    Args:
        on_save_callback: Function to call after saving settings
    """
    render_page_header("Settings", "Configure API keys and model settings for PaperChat")

    providers = get_available_providers()
    any_key_configured = any(get_api_key(provider) for provider in providers)

    if not any_key_configured:
        st.warning("No API keys configured. Please add at least one API key to use PaperChat.")

    render_settings_tabs(on_save_callback)


def render_settings_modal(on_close_callback: Callable | None = None):
    """
    Render settings as a modal overlay.

    This is a placeholder that would use Streamlit's experimental modal
    functionality when needed.

    Args:
        on_close_callback: Function to call when modal is closed
    """
    render_settings_page(on_close_callback)
