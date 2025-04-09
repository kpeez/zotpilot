"""
API Key Settings UI components for Streamlit.
"""

from typing import Callable

import streamlit as st

from paperchat.ui.common import show_error, show_info, show_success
from paperchat.utils.config import (
    get_api_key,
    get_available_providers,
    get_provider_display_name,
    mask_api_key,
    remove_api_key,
    set_api_key,
)
from paperchat.utils.settings import DEFAULT_PROVIDER

from ..llms.common import list_available_providers


def render_api_key_form(
    provider: str,
    on_save_callback: Callable | None = None,
    key_suffix: str = "",
) -> bool:
    """
    Render a form for entering and saving an API key.

    Args:
        provider: Provider name (e.g., "openai")
        on_save_callback: Function to call after successful save
        key_suffix: Optional suffix for form keys (needed for multiple forms on one page)

    Returns:
        True if key was saved successfully, False otherwise
    """
    provider_name = get_provider_display_name(provider)

    with st.form(key=f"{provider}_api_key_form{key_suffix}"):
        new_key = st.text_input(
            f"{provider_name} API Key",
            type="password",
            help=f"Enter your {provider_name} API key",
            placeholder=f"Enter {provider_name} API key here",
            key=f"{provider}_api_key_input{key_suffix}",
        )

        col1, col2 = st.columns([3, 1])
        with col1:
            submit = st.form_submit_button("Save API Key", use_container_width=True)

        if submit:
            if not new_key:
                show_error("API key cannot be empty.")
                return False

            success, error_msg = set_api_key(provider, new_key)
            if success:
                show_success(f"{provider_name} API key saved successfully!")
                if on_save_callback:
                    on_save_callback()
                return True
            else:
                show_error(f"Failed to save API key: {error_msg}")
                return False

    return False


def render_api_key_manager(
    provider: str,
    on_save_callback: Callable | None = None,
    show_header: bool = True,
    key_suffix: str = "",
) -> bool:
    """
    Render API key management UI for a specific provider.

    Args:
        provider: Provider name (e.g., "openai", "anthropic")
        on_save_callback: Function to call after successful save
        show_header: Whether to show the component header
        key_suffix: Optional suffix for component keys (useful for multiple instances)

    Returns:
        True if key is configured, False otherwise
    """
    provider_name = get_provider_display_name(provider)

    if show_header:
        st.subheader(f"{provider_name} API Key")

    api_key = get_api_key(provider)

    if api_key:
        masked_key = mask_api_key(api_key)
        st.success(f"✅ {provider_name} API key is configured")

        st.markdown(f"Current key: `{masked_key}`")
        if st.checkbox(f"Change {provider_name} API key", key=f"change_{provider}_key{key_suffix}"):
            return render_api_key_form(
                provider, on_save_callback, key_suffix=f"_change{key_suffix}"
            )

        if st.button(
            f"Remove {provider_name} API key",
            key=f"remove_{provider}_key{key_suffix}",
            help=f"Remove the {provider_name} API key from the system keyring",
        ):
            success, message = remove_api_key(provider)
            if success:
                show_success(message)
                if on_save_callback:
                    on_save_callback()
                st.rerun()  # Refresh the UI
            else:
                show_error(message)

        return True
    else:
        show_info(f"No {provider_name} API key configured")
        return render_api_key_form(provider, on_save_callback, key_suffix)


def render_api_keys_section(
    on_save_callback: Callable | None = None,
    providers: list[str] | None = None,
) -> None:
    """
    Render a section with API key management for all or specified providers.

    Args:
        on_save_callback: Function to call after saving any key
        providers: List of providers to show, or None for all available
    """
    if providers is None:
        providers = get_available_providers()

    for i, provider in enumerate(providers):
        if i > 0:
            st.divider()
        render_api_key_manager(provider, on_save_callback)


def render_api_key_setup():
    """Render the API key configuration interface."""
    st.header("🔑 API Keys Configuration")

    providers = list_available_providers()
    default_provider_index = 0
    if DEFAULT_PROVIDER in providers:
        default_provider_index = providers.index(DEFAULT_PROVIDER)

    selected_provider = st.selectbox(
        "Configure API Key for:",
        options=providers,
        index=default_provider_index,
    )

    current_api_key = get_api_key(selected_provider) or ""

    provider_info = {
        "openai": {
            "name": "OpenAI",
            "help": "Your OpenAI API key. Get your key at https://platform.openai.com/api-keys",
            "placeholder": "sk-...",
        },
        "anthropic": {
            "name": "Anthropic",
            "help": "Your Anthropic API key. Get your key at https://console.anthropic.com/",
            "placeholder": "sk-ant-...",
        },
    }

    info = provider_info.get(
        selected_provider,
        {
            "name": selected_provider.capitalize(),
            "help": f"Your {selected_provider.capitalize()} API key",
            "placeholder": "Enter your API key here...",
        },
    )

    st.write(f"Configure your {info['name']} API key to continue.")

    with st.form(key="api_key_form"):
        api_key = st.text_input(
            f"{info['name']} API Key",
            value=current_api_key,
            type="password",
            help=info["help"],
            placeholder=info["placeholder"],
        )

        submitted = st.form_submit_button("Save API Key")

        if submitted:
            if api_key:
                set_api_key(selected_provider, api_key)
                st.success(f"{info['name']} API key saved successfully!")
                st.session_state.show_api_setup = False
                st.rerun()
            else:
                st.error("Please enter a valid API key")

    st.subheader("API Key Status")

    for provider in providers:
        has_key = bool(get_api_key(provider))
        provider_display = provider_info.get(provider, {}).get("name", provider.capitalize())

        col1, col2, col3 = st.columns([3, 1, 1])
        with col1:
            st.write(f"{provider_display}")
        with col2:
            st.write("✅ Configured" if has_key else "❌ Not configured")
        with col3:
            if provider != selected_provider and st.button("Configure", key=f"config_{provider}"):
                st.session_state.show_api_setup = True
                st.experimental_rerun()

    if st.button("Continue to App"):
        st.session_state.show_api_setup = False
        st.rerun()
