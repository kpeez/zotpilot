"""
Settings UI components for Streamlit.
"""

from typing import Callable

import streamlit as st

from paperchat.ui.api_settings import render_api_key_manager
from paperchat.ui.common import render_page_header, show_info, show_warning
from paperchat.utils.config import get_api_key, get_available_providers

PROVIDER_MODELS = {
    "openai": ["gpt-4o", "gpt-4o-mini"],
    "anthropic": ["claude-3.7-sonnet", "claude-3.5-sonnet", "claude-3.5-haiku"],
    # TODO: add gemini models
}


def initialize_model_settings() -> str:
    """
    Initialize model settings in session state and return active provider.

    Returns:
        The active provider ID
    """

    if "provider_settings" not in st.session_state:
        st.session_state.provider_settings = {}

    if "active_provider" not in st.session_state:
        providers = get_available_providers()

        for provider in providers:
            if get_api_key(provider):
                st.session_state.active_provider = provider
                break

        if "active_provider" not in st.session_state:
            st.session_state.active_provider = providers[0] if providers else "openai"

    active_provider = st.session_state.active_provider

    if active_provider not in st.session_state.provider_settings:
        models = PROVIDER_MODELS.get(active_provider, [])
        st.session_state.provider_settings[active_provider] = {
            "model": models[0] if models else "",
            "temperature": 0.7,
        }

    return active_provider


def render_provider_selector() -> str:
    """
    Render provider selection section and return the selected provider.

    Returns:
        The selected provider ID
    """
    st.subheader("1️⃣ Provider")

    providers = get_available_providers()
    configured_providers = [p for p in providers if get_api_key(p)]

    if not configured_providers:
        show_warning("No API keys configured. Please add provider API keys in settings.")
        return st.session_state.active_provider

    selected_provider = st.radio(
        "Select AI Provider:",
        options=configured_providers,
        format_func=lambda x: x.capitalize(),
        index=configured_providers.index(st.session_state.active_provider)
        if st.session_state.active_provider in configured_providers
        else 0,
        key="provider_selector",
        horizontal=True,
    )

    if selected_provider != st.session_state.active_provider:
        st.session_state.active_provider = selected_provider
        st.rerun()

    return selected_provider


def render_model_selector(provider: str) -> str:
    """
    Render model selection dropdown for the specified provider.

    Args:
        provider: Provider ID

    Returns:
        The selected model ID
    """
    st.subheader("2️⃣ Model")

    models = PROVIDER_MODELS.get(provider, [])
    if not models:
        show_warning(f"No models configured for {provider.capitalize()}.")
        return ""

    current_settings = st.session_state.provider_settings.get(provider, {})
    current_model = current_settings.get("model", "")

    model_index = models.index(current_model) if current_model in models else 0

    selected_model = st.selectbox(
        f"Select {provider.capitalize()} Model:",
        options=models,
        index=model_index,
        key=f"{provider}_model_selector",
    )

    if "provider_settings" in st.session_state and provider in st.session_state.provider_settings:
        st.session_state.provider_settings[provider]["model"] = selected_model

    return selected_model


def render_parameter_settings(provider: str) -> float:
    """
    Render common parameter settings (temperature, etc).

    Args:
        provider: Provider ID

    Returns:
        The selected temperature value
    """
    st.subheader("3️⃣ Parameters")

    current_settings = st.session_state.provider_settings.get(provider, {})
    current_temp = current_settings.get("temperature", 0.7)

    temperature = st.slider(
        "Temperature",
        min_value=0.0,
        max_value=2.0,
        value=current_temp,
        step=0.1,
        help="Controls randomness in responses: 0=deterministic, higher=more creative",
        key=f"{provider}_temperature",
    )

    if "provider_settings" in st.session_state and provider in st.session_state.provider_settings:
        st.session_state.provider_settings[provider]["temperature"] = temperature

    return temperature


def update_global_settings(provider: str, model: str, temperature: float) -> None:
    """
    Update global settings based on provider selections.

    Args:
        provider: Selected provider ID
        model: Selected model ID
        temperature: Selected temperature value
    """
    if "settings" in st.session_state:
        st.session_state.settings.update(
            {
                "model": model,
                "temperature": temperature,
                "provider": provider,
            }
        )


def render_unified_model_settings() -> None:
    """
    Render a unified model settings panel with provider and model selection.

    This creates a single UI component with three sections:
    1. Provider selection (radio buttons)
    2. Model selection based on selected provider
    3. Common parameters (temperature)
    """
    initialize_model_settings()
    selected_provider = render_provider_selector()
    selected_model = render_model_selector(selected_provider)
    temperature = render_parameter_settings(selected_provider)
    update_global_settings(selected_provider, selected_model, temperature)


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

    st.slider(
        "Temperature",
        min_value=0.0,
        max_value=2.0,
        value=0.7,
        step=0.1,
        help="Controls randomness in responses: 0=deterministic, 1=creative",
        key=f"{provider}_temperature",
    )


def render_provider_settings(provider: str, on_save_callback: Callable | None = None) -> None:
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


def render_settings_tabs(on_save_callback: Callable | None = None) -> None:
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


def render_unified_settings_page(on_save_callback: Callable | None = None) -> None:
    """
    Render a settings page with unified model selection.

    Args:
        on_save_callback: Function to call after saving settings
    """
    render_page_header("Settings", "Configure API keys and model settings for PaperChat")

    with st.expander("API Keys", expanded=True):
        providers = get_available_providers()
        any_key_configured = any(get_api_key(provider) for provider in providers)

        if not any_key_configured:
            show_warning(
                "No API keys configured. Please add at least one API key to use PaperChat."
            )

        for provider in providers:
            with st.container():
                st.markdown(f"#### {provider.capitalize()}")
                render_api_key_manager(provider, on_save_callback, show_header=False)
                st.divider()

    with st.expander("Model Settings", expanded=True):
        render_unified_model_settings()


def render_settings_page(on_save_callback: Callable | None = None) -> None:
    """
    Render a complete settings page.

    Args:
        on_save_callback: Function to call after saving settings
    """
    render_unified_settings_page(on_save_callback)


def render_settings_modal(on_close_callback: Callable | None = None) -> None:
    """
    Render settings as a modal overlay.

    This is a placeholder that would use Streamlit's experimental modal
    functionality when needed.

    Args:
        on_close_callback: Function to call when modal is closed
    """
    render_settings_page(on_close_callback)
