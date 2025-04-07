"""
Settings UI components for Streamlit.
"""

from typing import Callable, Optional

import streamlit as st

from paperchat.ui.api_settings import render_api_key_manager
from paperchat.ui.common import render_page_header, show_info
from paperchat.utils.config import get_api_key, get_available_providers


def render_model_settings(provider: str):
    """
    Render model-specific settings UI for a provider.

    Args:
        provider: Provider name (e.g., "openai")
    """
    if provider == "openai":
        st.selectbox(
            "Default model",
            ["gpt-4o", "gpt-3.5-turbo"],
            help="Select which model to use by default",
            key="openai_default_model",
        )

        st.slider(
            "Temperature",
            min_value=0.0,
            max_value=2.0,
            value=0.7,
            step=0.1,
            help="Controls randomness in responses: 0=deterministic, 1=creative",
            key="openai_temperature",
        )

    elif provider == "anthropic":
        st.selectbox(
            "Default model",
            ["claude-3-opus", "claude-3-sonnet", "claude-3-haiku"],
            help="Select which model to use by default",
            key="anthropic_default_model",
        )

        st.slider(
            "Temperature",
            min_value=0.0,
            max_value=2.0,
            value=0.7,
            step=0.1,
            help="Controls randomness in responses: 0=deterministic, 1=creative",
            key="anthropic_temperature",
        )


def render_provider_settings(provider: str, on_save_callback: Optional[Callable] = None):
    """
    Render settings for a specific provider, including API key and model settings.

    Args:
        provider: Provider name (e.g., "openai")
        on_save_callback: Function to call after saving settings
    """
    # API key management
    api_key_configured = render_api_key_manager(provider, on_save_callback)

    # Only show model settings if API key is configured
    if api_key_configured:
        st.divider()
        st.subheader("Model Settings")
        render_model_settings(provider)

    # Add provider-specific help/links
    st.divider()
    if provider == "openai":
        st.markdown("[Get an OpenAI API key](https://platform.openai.com/api-keys)")
    elif provider == "anthropic":
        st.markdown("[Get an Anthropic API key](https://console.anthropic.com/)")


def render_settings_tabs(on_save_callback: Optional[Callable] = None):
    """
    Render settings as tabs, one for each provider.

    Args:
        on_save_callback: Function to call after saving settings
    """
    providers = get_available_providers()

    if not providers:
        show_info("No API providers configured.")
        return

    # Create tabs for each provider
    tab_labels = [provider.capitalize() for provider in providers]
    tabs = st.tabs(tab_labels)

    # Fill each tab with provider settings
    for i, provider in enumerate(providers):
        with tabs[i]:
            render_provider_settings(provider, on_save_callback)


def render_settings_page(on_save_callback: Optional[Callable] = None):
    """
    Render a complete settings page.

    Args:
        on_save_callback: Function to call after saving settings
    """
    render_page_header("Settings", "Configure API keys and model settings for PaperChat")

    # Check if any keys are configured
    providers = get_available_providers()
    any_key_configured = any(get_api_key(provider) for provider in providers)

    if not any_key_configured:
        st.warning("No API keys configured. Please add at least one API key to use PaperChat.")

    render_settings_tabs(on_save_callback)


def render_settings_modal(on_close_callback: Optional[Callable] = None):
    """
    Render settings as a modal overlay.

    This is a placeholder that would use Streamlit's experimental modal
    functionality when needed.

    Args:
        on_close_callback: Function to call when modal is closed
    """
    # This would use st.experimental_dialog or similar when available
    render_settings_page(on_close_callback)
