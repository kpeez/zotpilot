"""
Settings UI components for Streamlit.
"""

from typing import Callable

import streamlit as st

from paperchat.ui.api_settings import render_api_key_manager
from paperchat.ui.common import (
    render_page_header,
    show_error,
    show_info,
    show_success,
    show_warning,
)
from paperchat.utils.api_keys import get_api_key, get_available_providers

from ..llms.common import list_available_providers, list_models
from ..utils.settings import (
    DEFAULT_MAX_TOKENS,
    DEFAULT_MODEL,
    DEFAULT_PROVIDER,
    DEFAULT_TEMPERATURE,
)

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


def _handle_api_key_actions(provider: str) -> bool:
    """
    Handle API key actions like adding, changing, or removing a key.

    Args:
        provider: Provider name

    Returns:
        True if the UI needs to be rerun
    """
    is_configured = get_api_key(provider) is not None
    col1, col2, col3, col4 = st.columns([1, 1, 2, 1])

    with col1:
        st.markdown(f"**{provider.lower()}**")

    with col2:
        st.markdown(f"**{provider}**")

    with col3:
        if is_configured:
            from paperchat.utils.api_keys import mask_api_key

            masked_key = mask_api_key(get_api_key(provider))
            st.code(masked_key, language=None)
        else:
            st.markdown("🔑 *Not configured*")

    with col4:
        if is_configured:
            if st.button("Change", key=f"change_{provider}_btn"):
                st.session_state[f"show_{provider}_form"] = True

            if st.button("Remove", key=f"remove_{provider}_btn"):
                from paperchat.utils.api_keys import remove_api_key

                success, message = remove_api_key(provider)
                if success:
                    show_success(message)
                    return True
                else:
                    show_error(message)
        elif st.button("Add Key", key=f"add_{provider}_btn"):
            st.session_state[f"show_{provider}_form"] = True

    if st.session_state.get(f"show_{provider}_form", False):
        with st.expander(f"Configure {provider.capitalize()} API Key", expanded=True):
            from paperchat.ui.api_settings import render_api_key_form

            if render_api_key_form(provider, on_save_callback=lambda: st.rerun()):
                st.session_state[f"show_{provider}_form"] = False
                return True

            if st.button("Cancel", key=f"cancel_{provider}_form"):
                st.session_state[f"show_{provider}_form"] = False
                return True

    return False


def render_compact_api_key_table() -> None:
    """
    Render a compact table of API providers with status and actions.

    This creates a table-like UI with columns:
    - ID: Provider name
    - Type: Provider type (LLM/embeddings)
    - API Key: Shows masked key if configured or "Not configured"
    - Actions: Buttons to add/remove keys
    """
    providers = get_available_providers()
    if not providers:
        show_info("No API providers are configured in the system.")
        return

    st.markdown("### API Providers")
    col1, col2, col3, col4 = st.columns([1, 1, 2, 1])
    with col1:
        st.markdown("**ID**")
    with col2:
        st.markdown("**Type**")
    with col3:
        st.markdown("**API Key**")
    with col4:
        st.markdown("**Actions**")

    st.divider()

    for provider in providers:
        rerun_needed = _handle_api_key_actions(provider)
        if rerun_needed:
            st.rerun()
        st.divider()


def render_model_selection_table() -> None:
    """
    Render a compact table of model settings for the active provider.
    """
    active_provider = initialize_model_settings()

    models = PROVIDER_MODELS.get(active_provider, [])
    if not models:
        show_warning(f"No models configured for {active_provider.capitalize()}.")
        return

    st.markdown(f"### {active_provider.capitalize()} Models")

    st.radio(
        "Provider:",
        options=[p for p in get_available_providers() if get_api_key(p)],
        format_func=lambda x: x.capitalize(),
        index=0,
        key="provider_table_selector",
        horizontal=True,
        on_change=lambda: st.session_state.update(
            {"active_provider": st.session_state.provider_table_selector}
        ),
    )

    current_settings = st.session_state.provider_settings.get(active_provider, {})
    current_model = current_settings.get("model", models[0])

    selected_model = st.selectbox(
        "Model:",
        options=models,
        index=models.index(current_model) if current_model in models else 0,
        key=f"{active_provider}_model_table_selector",
    )

    temperature = st.slider(
        "Temperature:",
        min_value=0.0,
        max_value=2.0,
        value=current_settings.get("temperature", 0.7),
        step=0.1,
        help="Controls randomness in responses: 0=deterministic, higher=more creative",
    )

    if active_provider in st.session_state.provider_settings:
        st.session_state.provider_settings[active_provider].update(
            {"model": selected_model, "temperature": temperature}
        )

    update_global_settings(active_provider, selected_model, temperature)


def render_unified_settings_page(on_save_callback: Callable | None = None) -> None:
    """
    Render a settings page with unified model selection.

    Args:
        on_save_callback: Function to call after saving settings
    """
    render_page_header("Settings", "Configure API keys and model settings for PaperChat")

    with st.expander("API Keys", expanded=True):
        render_compact_api_key_table()

    with st.expander("Model Selection", expanded=True):
        render_model_selection_table()

    with st.expander("Retrieval Settings", expanded=True):
        st.subheader("Document Retrieval")
        top_k = st.slider(
            "Number of chunks to retrieve",
            min_value=1,
            max_value=10,
            value=st.session_state.settings.get("top_k", 5),
            help="Higher values retrieve more document chunks but may include less relevant information",
        )

        if "settings" in st.session_state:
            st.session_state.settings.update({"top_k": top_k})


def render_settings_page(on_save_callback: Callable | None = None) -> None:
    """
    Render the settings configuration page.

    Args:
        on_save_callback: Optional callback to execute after settings are saved
    """
    st.header("⚙️ Settings")
    st.write("Configure your chat experience")

    settings = st.session_state.settings

    with st.form("settings_form"):
        st.subheader("Model Settings")
        providers = list_available_providers()
        provider = st.selectbox(
            "LLM Provider",
            options=providers,
            index=providers.index(settings.get("provider", DEFAULT_PROVIDER))
            if settings.get("provider") in providers
            else 0,
            help="Select which LLM provider to use",
        )
        provider_models = list_models(provider_name=provider)
        model_ids = [model["id"] for model in provider_models]
        current_model = settings.get("model", DEFAULT_MODEL)
        model_index = 0
        if current_model in model_ids:
            model_index = model_ids.index(current_model)

        model = st.selectbox(
            "Model", options=model_ids, index=model_index, help="Select which language model to use"
        )

        temperature = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=2.0,
            value=settings.get("temperature", DEFAULT_TEMPERATURE),
            step=0.05,
            format="%.2f",
            help="Controls randomness in the model. Lower values are more deterministic, higher values more creative.",
        )

        max_tokens = st.number_input(
            "Max response tokens",
            min_value=100,
            max_value=4000,
            value=settings.get("max_tokens", DEFAULT_MAX_TOKENS),
            step=50,
            help="Maximum number of tokens (words/word pieces) in the model's response.",
        )

        st.subheader("Retrieval Settings")
        top_k = st.slider(
            "Number of context chunks",
            min_value=1,
            max_value=20,
            value=settings.get("top_k", 5),
            step=1,
            help="How many chunks from the document to include in the model's context.",
        )

        save_button = st.form_submit_button("Save Settings")

        if save_button:
            settings["provider"] = provider
            settings["model"] = model
            settings["temperature"] = temperature
            settings["max_tokens"] = max_tokens
            settings["top_k"] = top_k
            # update the LLM client with the new provider if it changed
            if provider != st.session_state.get("last_provider", DEFAULT_PROVIDER):
                from ..llms.common import get_client

                st.session_state.llm_client = get_client(provider_name=provider)
                st.session_state["last_provider"] = provider

            st.success("Settings saved!")

            if on_save_callback:
                on_save_callback()


def render_settings_modal(on_close_callback: Callable | None = None) -> None:
    """
    Render settings as a modal overlay.

    This is a placeholder that would use Streamlit's experimental modal
    functionality when needed.

    Args:
        on_close_callback: Function to call when modal is closed
    """
    render_settings_page(on_close_callback)
