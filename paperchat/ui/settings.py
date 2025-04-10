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
from paperchat.utils.api_keys import (
    get_api_key,
    get_available_providers,
    get_provider_display_name,
    mask_api_key,
    remove_api_key,
    set_api_key,
)

from ..llms.common import list_models


def add_api_table_css() -> None:
    """
    Add custom CSS for the API key table styling.
    """
    st.markdown(
        """
    <style>
    /* Code block styling for masked keys */
    pre {
        padding: 0.5rem !important;
        background-color: #f0f0f0 !important;
        border-radius: 4px !important;
        margin: 0 !important;
    }
    .stDivider {
        margin-top: 0.5rem !important;
        margin-bottom: 0.5rem !important;
    }
    </style>
    """,
        unsafe_allow_html=True,
    )


def handle_api_key_popup(provider: str) -> None:
    """
    Handle the API key popup interaction for a provider.

    Args:
        provider: Provider ID (e.g., 'openai')
    """
    provider_name = get_provider_display_name(provider)
    current_key = get_api_key(provider) or ""

    with st.form(key=f"{provider}_popup_form"):
        api_key = st.text_input(
            label="Edit Provider API Key",
            value=current_key,
            type="password" if current_key else "default",
            placeholder=f"Enter {provider_name} API key...",
            key=f"{provider}_popup_input",
        )

        col1, col2 = st.columns(2)
        with col1:
            save = st.form_submit_button("Save", use_container_width=True)
        with col2:
            if current_key:
                delete = st.form_submit_button("Delete", use_container_width=True)
            else:
                cancel = st.form_submit_button("Cancel", use_container_width=True)

        if save and api_key:
            success, error_msg = set_api_key(provider, api_key)
            if success:
                show_success(f"{provider_name} API key saved successfully!")
                st.session_state[f"show_{provider}_popup"] = False
                st.rerun()
            else:
                show_error(f"Failed to save API key: {error_msg}")

        if current_key and delete:
            success, message = remove_api_key(provider)
            if success:
                show_success(message)
                st.session_state[f"show_{provider}_popup"] = False
                st.rerun()
            else:
                show_error(message)

        if not current_key and cancel:
            st.session_state[f"show_{provider}_popup"] = False
            st.rerun()


def render_model_table() -> None:
    """
    Render a table of all available models across providers.
    """
    st.markdown("#### Available Models")

    all_models = []
    for provider in get_available_providers():
        if not get_api_key(provider):
            continue

        models = list_models(provider_name=provider)
        for model in models:
            all_models.append(
                {
                    "provider": provider,
                    "provider_display": get_provider_display_name(provider),
                    "model_id": model["id"],
                }
            )

    if not all_models:
        show_info("No models available. Please configure API keys first.")
        return

    col1, col2, col3 = st.columns([1, 2, 1])
    with col1:
        st.markdown("**Provider**")
    with col2:
        st.markdown("**Model ID**")
    with col3:
        st.markdown("**Status**")

    st.divider()

    active_provider = st.session_state.get("active_provider", "")
    active_model = ""
    if active_provider in st.session_state.get("provider_settings", {}):
        active_model = st.session_state.provider_settings[active_provider].get("model", "")

    for model_info in all_models:
        col1, col2, col3 = st.columns([1, 2, 1])

        with col1:
            st.markdown(f"{model_info['provider_display']}")

        with col2:
            st.markdown(f"`{model_info['model_id']}`")

        with col3:
            is_selected = (
                model_info["provider"] == active_provider and model_info["model_id"] == active_model
            )

            if is_selected:
                st.markdown("✅ **Active**")
            elif st.button(
                "Select", key=f"select_{model_info['provider']}_{model_info['model_id']}"
            ):
                st.session_state.active_provider = model_info["provider"]

                if model_info["provider"] not in st.session_state.provider_settings:
                    st.session_state.provider_settings[model_info["provider"]] = {
                        "temperature": 0.7
                    }

                st.session_state.provider_settings[model_info["provider"]]["model"] = model_info[
                    "model_id"
                ]

                update_global_settings(
                    model_info["provider"],
                    model_info["model_id"],
                    st.session_state.provider_settings[model_info["provider"]].get(
                        "temperature", 0.7
                    ),
                )

                show_success(f"Selected {model_info['model_id']} as active model")
                st.rerun()

        st.divider()


def render_api_key_table() -> None:
    """
    Render a table of API providers using Streamlit's column layout for a native look.
    This approach allows embedding interactive buttons directly in the table.
    """
    providers = get_available_providers()

    if not providers:
        show_info("No API providers are configured.")
        return

    col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
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
        col1, col2, col3, col4 = st.columns([1, 1, 1, 1])

        with col1:
            st.markdown(f"{provider.lower()}")

        with col2:
            st.markdown(f"{get_provider_display_name(provider)}")

        with col3:
            api_key = get_api_key(provider)
            if api_key:
                st.code(mask_api_key(api_key), language=None)
            else:
                st.markdown("No API Key set")

        with col4, st.popover("⚙️"):
            handle_api_key_popup(provider)

        st.divider()


def render_model_selector(provider: str) -> str:
    """
    Render model selection dropdown for the specified provider.

    Args:
        provider: Provider ID

    Returns:
        The selected model ID
    """
    st.subheader("2️⃣ Model")

    models = list_models(provider_name=provider)
    model_ids = [model["id"] for model in models]

    if not model_ids:
        show_warning(f"No models configured for {provider.capitalize()}.")
        return ""

    current_settings = st.session_state.provider_settings.get(provider, {})
    current_model = current_settings.get("model", "")

    model_index = model_ids.index(current_model) if current_model in model_ids else 0

    selected_model = st.selectbox(
        f"Select {provider.capitalize()} Model:",
        options=model_ids,
        index=model_index,
        key=f"{provider}_model_selector",
    )

    if "provider_settings" in st.session_state and provider in st.session_state.provider_settings:
        st.session_state.provider_settings[provider]["model"] = selected_model

    return selected_model


def render_model_settings(provider: str) -> None:
    """
    Render model-specific settings UI for a provider.

    Args:
        provider: Provider name (e.g., "openai")
    """
    provider = provider.lower()
    models = list_models(provider_name=provider)
    model_ids = [model["id"] for model in models]

    if model_ids:
        st.selectbox(
            "Default model",
            model_ids,
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


def render_model_selection_table() -> None:
    """
    Render a compact table of model settings for the active provider.
    """
    active_provider = initialize_model_settings()

    models = list_models(provider_name=active_provider)
    model_ids = [model["id"] for model in models]

    if not model_ids:
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
    current_model = current_settings.get("model", model_ids[0] if model_ids else "")

    selected_model = st.selectbox(
        "Model:",
        options=model_ids,
        index=model_ids.index(current_model) if current_model in model_ids else 0,
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
        models = list_models(provider_name=active_provider)
        model_ids = [model["id"] for model in models]

        st.session_state.provider_settings[active_provider] = {
            "model": model_ids[0] if model_ids else "",
            "temperature": 0.7,
        }

    return active_provider


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


def render_unified_settings_page(on_save_callback: Callable | None = None) -> None:
    """
    Render a settings page with unified model selection.

    Args:
        on_save_callback: Function to call after saving settings
    """
    render_page_header("Settings", "Configure API keys and model settings for PaperChat")

    with st.expander("Model Selection", expanded=True):
        render_model_table()

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
    add_api_table_css()
    st.header("⚙️ Settings")
    if st.button("Back to Chat", icon="⬅️", type="tertiary", use_container_width=False):
        st.session_state.show_settings = False
        st.rerun()

    st.subheader("Providers")
    st.caption("Configure API keys for each provider.")
    render_api_key_table()

    st.subheader("Models")
    st.caption("Select the models to use.")
    render_model_table()

    st.subheader("Model & Retrieval Settings")
    with st.form("settings_form"):
        active_provider = st.session_state.get("active_provider", "")
        active_settings = st.session_state.provider_settings.get(active_provider, {})
        active_model = active_settings.get("model", "")

        st.markdown(
            f"**Current Model:** {get_provider_display_name(active_provider)} - `{active_model}`"
        )

        temperature = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=2.0,
            value=active_settings.get("temperature", 0.7),
            step=0.1,
            help="Controls randomness in responses: 0=deterministic, higher=more creative",
        )

        st.markdown("### Retrieval Settings")
        top_k = st.slider(
            "Number of chunks to retrieve",
            min_value=1,
            max_value=20,
            value=st.session_state.settings.get("top_k", 5),
            step=1,
            help="How many chunks from the document to include in the model's context.",
        )

        save_button = st.form_submit_button("Save Settings")

        if save_button:
            if active_provider in st.session_state.provider_settings:
                st.session_state.provider_settings[active_provider]["temperature"] = temperature

            st.session_state.settings["top_k"] = top_k

            if active_provider and active_model:
                update_global_settings(active_provider, active_model, temperature)

            show_success("Settings saved!")
