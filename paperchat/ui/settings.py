"""
Settings UI components for Streamlit.
"""

from typing import Callable

import streamlit as st

from paperchat.core.embeddings import (
    PROVIDERS_REQUIRING_KEYS,
    list_embedding_models,
)
from paperchat.ui.common import render_page_header
from paperchat.utils.api_keys import (
    get_api_key,
    get_available_providers,
    get_provider_display_name,
    mask_api_key,
    remove_api_key,
    set_api_key,
)
from paperchat.utils.settings import get_active_embedding_model, set_active_embedding_model

from ..llms.common import list_llm_models


def add_api_table_css() -> None:
    """
    Add custom CSS for the API key table styling.
    """
    st.markdown(
        """
    <style>
    /* Default (light mode) code block styling for masked keys */
    /* Applied directly to the pre element used by st.code */
    .stCodeBlock pre {
        padding: 0.5rem !important;
        background-color: #f0f0f0 !important; /* Light background */
        color: #000000 !important; /* Black text */
        border-radius: 4px !important;
        margin: 0 !important;
        font-family: 'SFMono-Regular', Consolas, 'Liberation Mono', Menlo, Courier, monospace !important;
        font-size: 0.9em !important;
        line-height: 1.2 !important;
    }

    /* Dark mode override for the masked key code blocks */
    @media (prefers-color-scheme: dark) {
        .stCodeBlock pre {
            background-color: #262730 !important; /* Dark background */
            color: #ffffff !important; /* White text */
            /* Inherits padding, border-radius, margin, font */
        }
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
                close_popover = st.form_submit_button("Close", use_container_width=True)

        if save and api_key:
            success, error_msg = set_api_key(provider, api_key)
            if success:
                st.success(f"{provider_name} API key saved successfully!")
                st.rerun()
            else:
                st.error(f"Failed to save API key: {error_msg}")

        if current_key and delete:
            success, message = remove_api_key(provider)
            if success:
                st.success(message)
                st.rerun()
            else:
                st.error(message)

        if not current_key and close_popover:
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

        models = list_llm_models(provider_name=provider)
        for model in models:
            all_models.append(
                {
                    "provider": provider,
                    "provider_display": get_provider_display_name(provider),
                    "model_id": model["id"],
                }
            )

    if not all_models:
        st.info("No models available. Please configure API keys first.")
        return

    active_provider = st.session_state.config.get("provider_name", "")
    active_model = st.session_state.config.get("model_id", "")

    col1, col2, col3 = st.columns([2, 3, 2])
    with col1:
        st.markdown("**Provider**")
    with col2:
        st.markdown("**Model ID**")
    with col3:
        st.markdown("**Status**")

    st.divider()

    for model_info in all_models:
        col1, col2, col3 = st.columns([2, 3, 2])

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
                st.session_state.config["provider_name"] = model_info["provider"]
                st.session_state.config["model_id"] = model_info["model_id"]

                st.success(f"Selected {model_info['model_id']} as active model")
                st.rerun()

        st.divider()


def render_api_key_table() -> None:
    """
    Render a table of API providers using Streamlit's column layout for a native look.
    This approach allows embedding interactive buttons directly in the table.
    """
    providers = get_available_providers()

    if not providers:
        st.info("No API providers are configured.")
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

        with col4, st.popover("⚙️", use_container_width=False):
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

    models = list_llm_models(provider_name=provider)
    model_ids = [model["id"] for model in models]

    if not model_ids:
        st.warning(f"No models configured for {provider.capitalize()}.")
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
    models = list_llm_models(provider_name=provider)
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
        st.warning("No API keys configured. Please add provider API keys in settings.")
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
    if "config" not in st.session_state:
        st.session_state.config = {
            "provider_name": provider,
            "model_id": model,
            "temperature": temperature,
            "max_tokens": 1000,
            "top_k": 5,
        }
    else:
        st.session_state.config.update(
            {
                "model_id": model,
                "temperature": temperature,
                "provider_name": provider,
            }
        )


def _render_llm_settings_form_section() -> tuple[str | None, str | None, float]:
    """Renders the LLM provider, model, and temperature selectors."""
    st.markdown("### LLM Configuration")
    available_llm_providers = get_available_providers()
    selected_model_id = None
    selected_temperature = 0.7

    if not available_llm_providers:
        st.warning("No LLM providers available. Please configure API keys first.")
        current_llm_provider = None
    else:
        current_llm_provider_name = st.session_state.config.get(
            "provider_name", get_available_providers()[0]
        )
        if current_llm_provider_name not in available_llm_providers:
            current_llm_provider_name = available_llm_providers[0]

        llm_provider_index = available_llm_providers.index(current_llm_provider_name)
        current_llm_provider = st.selectbox(
            "LLM Provider",
            options=available_llm_providers,
            index=llm_provider_index,
            format_func=get_provider_display_name,
            key="llm_provider_selector",
            help="Select the Large Language Model provider.",
        )

    if current_llm_provider:
        models = list_llm_models(provider_name=current_llm_provider)
        model_ids = [m["id"] for m in models]
        current_model_id = st.session_state.config.get("model_id", "")
        if current_model_id not in model_ids:
            current_model_id = model_ids[0] if model_ids else ""
        model_index = model_ids.index(current_model_id) if current_model_id in model_ids else 0

        selected_model_id = st.selectbox(
            "Model",
            options=model_ids,
            index=model_index,
            key="model_selector",
            help="Select the specific model to use.",
            disabled=not model_ids,
        )

        current_temperature = st.session_state.config.get("temperature", 0.7)
        selected_temperature = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=1.0,
            value=current_temperature,
            step=0.05,
            key="temperature_slider",
            help="Controls randomness. Lower values make responses more deterministic.",
        )
    else:
        st.selectbox("Model", options=[], disabled=True)
        st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.7, disabled=True)

    return current_llm_provider, selected_model_id, selected_temperature


def _render_embedding_settings_form_section() -> str | None:
    """Renders the Embedding model selector."""
    st.markdown("### Embedding Configuration")
    available_embed_models = list_embedding_models()
    embed_model_ids = [m["id"] for m in available_embed_models]
    selected_embed_model_id = None

    if not embed_model_ids:
        st.warning(
            "No Embedding models available. Configure required API keys (e.g., Gemini or OpenAI) or rely on the offline default."
        )
        st.info("Offline default: `milvus/all-MiniLM-L6-v2` will be used.")
        current_embed_model_id = "milvus/all-MiniLM-L6-v2"
        st.selectbox(
            "Embedding Model",
            options=[current_embed_model_id],
            index=0,
            key="embedding_model_selector",
            help="Select the model used for document embedding and search. Requires API keys for some models.",
            disabled=True,
        )
    else:
        current_embed_model_id = get_active_embedding_model()
        if current_embed_model_id not in embed_model_ids:
            embed_model_index = 0
            current_embed_model_id = embed_model_ids[0]
        else:
            embed_model_index = embed_model_ids.index(current_embed_model_id)

        selected_embed_model_id = st.selectbox(
            "Embedding Model",
            options=embed_model_ids,
            index=embed_model_index,
            key="embedding_model_selector",
            help="Select the model used for document embedding and search. API keys required for OpenAI/Gemini models.",
        )
    return selected_embed_model_id


def _save_llm_settings(
    selected_llm_provider: str | None,
    selected_llm_model: str | None,
    selected_llm_temp: float,
) -> bool:
    """Saves LLM settings to session state and returns True if changed."""
    changed = False
    if selected_llm_provider and selected_llm_model:
        if st.session_state.config.get("provider_name") != selected_llm_provider:
            st.session_state.config["provider_name"] = selected_llm_provider
            changed = True
        if st.session_state.config.get("model_id") != selected_llm_model:
            st.session_state.config["model_id"] = selected_llm_model
            changed = True
        if st.session_state.config.get("temperature") != selected_llm_temp:
            st.session_state.config["temperature"] = selected_llm_temp
            changed = True

        if changed:
            st.success(
                f"LLM settings saved: Provider '{selected_llm_provider}', Model '{selected_llm_model}', Temp {selected_llm_temp:.2f}"
            )
        return changed  # Return True if anything was actually changed
    else:
        st.warning("LLM Provider/Model not fully selected. LLM settings not saved.")
        return False


def _save_embedding_settings(selected_embed_model: str | None) -> bool | None:
    """Saves embedding setting after checking API key. Returns True on success, False on key error, None if no model selected."""
    if selected_embed_model is None:
        return None  # Indicate no action taken

    provider = selected_embed_model.split("/")[0]
    key_needed = provider in PROVIDERS_REQUIRING_KEYS
    key_present = get_api_key(provider) is not None

    if key_needed and not key_present:
        st.error(
            f"Cannot save: Embedding model '{selected_embed_model}' requires an API key for '{provider}', which is missing."
        )
        return False  # Indicate failure due to missing key

    if get_active_embedding_model() != selected_embed_model:
        save_attempt = set_active_embedding_model(selected_embed_model)
        if save_attempt:
            st.success(f"Embedding model saved: '{selected_embed_model}'")
            return True  # Indicate successful save
        else:
            st.error("Failed to save embedding model setting.")
            return False  # Indicate failure during save
    else:
        return True  # Indicate success (no change needed)


def _handle_settings_form_submission(
    selected_llm_provider: str | None,
    selected_llm_model: str | None,
    selected_llm_temp: float,
    selected_embed_model: str | None,
    on_save_callback: Callable | None,
) -> None:
    """Handles the logic after the settings form is submitted."""
    llm_settings_changed = _save_llm_settings(
        selected_llm_provider, selected_llm_model, selected_llm_temp
    )
    embedding_save_status = _save_embedding_settings(selected_embed_model)

    # --- Callback and Rerun Logic ---
    # Callback if *any* setting was successfully changed or saved (even if value didn't change)
    if (llm_settings_changed or embedding_save_status is True) and on_save_callback:
        on_save_callback()

    # Rerun if *any* setting was successfully changed or saved,
    # *unless* the only action was an embedding save failure (due to key error).
    if llm_settings_changed or embedding_save_status is True:
        st.rerun()
    # No explicit else needed, as we don't rerun on embedding_save_status == False or None
    # if llm_settings_changed is also False.


def render_unified_settings_page(on_save_callback: Callable | None = None) -> None:
    """Render the unified settings page with LLM and Embedding settings."""
    if st.button("← Back"):
        st.session_state.show_settings = False
        st.rerun()

    render_page_header("Application Settings", "Configure API Keys, LLM, and Embedding models.")
    add_api_table_css()

    st.markdown("### API Key Management")
    st.info(
        "Enter API keys for the providers you wish to use. Keys are stored securely.", icon="🔑"
    )
    render_api_key_table()

    st.divider()

    # LLM and Embedding settings within a form
    with st.form(key="unified_settings_form"):
        col1, col2 = st.columns(2)
        with col1:
            # Render LLM settings section
            selected_llm_provider, selected_llm_model, selected_llm_temp = (
                _render_llm_settings_form_section()
            )
        with col2:
            # Render Embedding settings section
            selected_embed_model = _render_embedding_settings_form_section()

        submitted = st.form_submit_button("Save Settings", use_container_width=True)

        if submitted:
            # Handle form submission logic
            _handle_settings_form_submission(
                selected_llm_provider,
                selected_llm_model,
                selected_llm_temp,
                selected_embed_model,
                on_save_callback,
            )


def render_settings_page(on_save_callback: Callable | None = None) -> None:
    """Main function to render the settings page content."""
    render_unified_settings_page(on_save_callback)
