"""
API key settings UI components for Streamlit.
"""

from typing import Callable, Optional

import streamlit as st

from paperchat.utils.api_keys import (
    get_api_key,
    get_provider_display_name,
    remove_api_key,
    set_api_key,
)


def render_api_key_form(provider: str, on_save_callback: Optional[Callable] = None) -> bool:
    """
    Render an API key configuration form for a provider and handle the form submission.

    Args:
        provider: Provider ID (e.g., 'openai')
        on_save_callback: Function to call when settings are saved

    Returns:
        True if form was submitted, False otherwise
    """
    provider_name = get_provider_display_name(provider)

    form_key = f"{provider}_api_key_form"

    with st.form(form_key):
        api_key = st.text_input(
            f"{provider_name} API Key",
            type="password",
            value=get_api_key(provider) or "",
            placeholder=f"Enter your {provider_name} API key",
            help=f"API key for {provider_name} services. This will be securely stored on your device.",
        )

        submitted = st.form_submit_button("Save")

        if submitted:
            # Check for empty value
            if not api_key:
                st.error("API key cannot be empty.")
                return False

            # Save the API key
            success, error_msg = set_api_key(provider, api_key)
            if success:
                st.success(f"{provider_name} API key saved successfully!")
                if on_save_callback:
                    on_save_callback()
                return True
            else:
                st.error(f"Failed to save API key: {error_msg}")
                return False

    return False


def render_api_key_section(provider: str, on_save_callback: Optional[Callable] = None) -> bool:
    """
    Render a provider API key section with current status and edit form.

    Args:
        provider: Provider ID
        on_save_callback: Function to call when settings are saved

    Returns:
        True if API key is configured, False otherwise
    """
    provider_name = get_provider_display_name(provider)
    api_key = get_api_key(provider)
    is_configured = api_key is not None

    col1, col2 = st.columns([3, 1])

    with col1:
        if is_configured:
            st.success(f"{provider_name} API key is configured")
        else:
            st.info(f"No {provider_name} API key configured")

    with col2:
        if is_configured and st.button(f"Remove {provider} Key", key=f"remove_{provider}_key"):
            success, message = remove_api_key(provider)
            if success:
                st.success(message)
                if on_save_callback:
                    on_save_callback()
                st.rerun()
            else:
                st.error(message)
                return is_configured

    # Render the form if not configured or edit button pressed
    if not is_configured or st.button(f"Edit {provider} Key", key=f"edit_{provider}_key"):
        render_api_key_form(provider, on_save_callback)

    return is_configured


def render_api_key_manager(provider: str, on_save_callback: Optional[Callable] = None) -> bool:
    """
    Render a complete API key management UI for a provider.

    Args:
        provider: Provider ID (e.g., 'openai')
        on_save_callback: Function to call when settings are saved

    Returns:
        True if API key is configured, False otherwise
    """
    provider_name = get_provider_display_name(provider)
    api_key = get_api_key(provider)
    is_configured = api_key is not None

    st.subheader(f"{provider_name} API Key")

    if is_configured:
        st.success(f"{provider_name} API key is configured")

        col1, col2 = st.columns(2)
        with col1:
            if st.button("Edit API Key", key=f"edit_{provider}_api_key"):
                st.session_state[f"show_{provider}_edit"] = True
        with col2:
            if st.button("Remove API Key", key=f"remove_{provider}_api_key"):
                success, message = remove_api_key(provider)
                if success:
                    st.success(message)
                    if on_save_callback:
                        on_save_callback()
                    st.rerun()
                else:
                    st.error(message)
    else:
        st.info(f"No {provider_name} API key configured")
        st.session_state[f"show_{provider}_edit"] = True

    if st.session_state.get(f"show_{provider}_edit", False):
        render_api_key_form(provider, on_save_callback)

    return is_configured


def render_api_keys_section(on_save_callback: Optional[Callable] = None) -> None:
    """
    Render a section showing all available API providers and their configuration status.

    Args:
        on_save_callback: Function to call when settings are saved
    """
    from paperchat.utils.api_keys import get_available_providers

    st.subheader("API Keys")
    st.markdown("Configure API keys for language model providers:")

    for provider in get_available_providers():
        with st.expander(get_provider_display_name(provider), expanded=True):
            render_api_key_section(provider, on_save_callback)
            if provider == "openai":
                st.caption(
                    "Get an OpenAI API key from [OpenAI API Keys](https://platform.openai.com/api-keys)"
                )
            elif provider == "anthropic":
                st.caption(
                    "Get an Anthropic API key from [Anthropic Console](https://console.anthropic.com/)"
                )
            elif provider == "gemini":
                st.caption(
                    "Get a Gemini API key from [Google AI Studio](https://aistudio.google.com/app/apikey)"
                )

        st.divider()
