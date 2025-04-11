"""
Document handling UI components for Streamlit.
"""

import hashlib
import os
import tempfile

import streamlit as st

from paperchat.core import process_document
from paperchat.ui.common import refresh_model_state
from paperchat.ui.settings import (
    initialize_model_settings,
    update_global_settings,
)


def render_compact_settings_ui() -> None:
    """Render a compact settings UI for the sidebar."""
    initialize_model_settings()

    from paperchat.llms.common import list_models
    from paperchat.utils.api_keys import (
        get_api_key,
        get_available_providers,
    )

    all_models = []
    active_provider = st.session_state.config.get("provider_name", "")
    active_model = st.session_state.config.get("model_id", "")

    for provider in get_available_providers():
        if not get_api_key(provider):
            continue

        models = list_models(provider_name=provider)
        for model in models:
            model_display = f"{model['id']}"
            all_models.append(
                {"display": model_display, "provider": provider, "model_id": model["id"]}
            )

    if not all_models:
        st.warning("No models available. Please configure API keys in settings.")
        return

    st.markdown("**Model Selection**")

    selected_index = 0
    for i, model_info in enumerate(all_models):
        if model_info["provider"] == active_provider and model_info["model_id"] == active_model:
            selected_index = i
            break

    selected_model = st.selectbox(
        "Select model:",
        options=range(len(all_models)),
        format_func=lambda i: all_models[i]["display"],
        index=selected_index,
    )

    selected_provider = all_models[selected_model]["provider"]
    selected_model_id = all_models[selected_model]["model_id"]

    if selected_provider != active_provider or selected_model_id != active_model:
        st.session_state.config.update(
            {
                "provider_name": selected_provider,
                "model_id": selected_model_id,
            }
        )

    current_temp = st.session_state.config.get("temperature", 0.7)
    temperature = st.slider(
        "Temperature",
        min_value=0.0,
        max_value=2.0,
        value=current_temp,
        step=0.1,
        help="Controls randomness: 0=deterministic, 2=creative",
    )

    st.session_state.config["temperature"] = temperature

    st.markdown("---")
    st.markdown("**Retrieval settings:**")
    top_k = st.slider(
        "Chunks to retrieve",
        min_value=1,
        max_value=10,
        value=st.session_state.config.get("top_k", 5),
        help="Higher values retrieve more chunks but may include less relevant information",
    )

    update_global_settings(selected_provider, selected_model_id, temperature)

    st.session_state.config["top_k"] = top_k


def render_upload_section() -> None:
    """Render document upload section in the sidebar."""
    st.markdown("Upload a PDF to start chatting with its content.")

    uploaded_file = st.file_uploader(
        "Choose a PDF file", type=["pdf"], help="Select a PDF file to upload and process"
    )

    if uploaded_file is not None:
        original_filename = uploaded_file.name
        file_content = uploaded_file.getvalue()
        content_hash = hashlib.md5(file_content).hexdigest()

        if content_hash in st.session_state.processed_documents:
            st.success(f"Document already processed: {original_filename}")
            if st.session_state.active_document_hash != content_hash:
                st.session_state.active_document_hash = content_hash
                st.session_state.messages = []
                st.rerun()
        else:
            progress_bar = st.progress(0)
            st.markdown("⏳ Processing document... This may take a moment.")

            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                    tmp_file.write(file_content)
                    pdf_path = tmp_file.name

                progress_bar.progress(10, "Preparing document...")
                progress_bar.progress(20, "Parsing PDF document...")
                progress_bar.progress(40, "Extracting text content...")

                document_data = process_document(
                    pdf_path, embedding_model=st.session_state.embedding_model
                )
                progress_bar.progress(70, "Creating embeddings...")
                progress_bar.progress(90, "Finalizing...")

                st.session_state.processed_documents[content_hash] = document_data
                st.session_state.filenames[content_hash] = original_filename
                st.session_state.active_document_hash = content_hash
                os.unlink(pdf_path)

                progress_bar.progress(100, "Complete!")
                st.success(f"Document processed: {original_filename}")
                st.session_state.messages = []
                st.rerun()
            except Exception as e:
                progress_bar.progress(100, "Error!")
                st.error(f"Error processing document: {e}")
                if "pdf_path" in locals():
                    os.unlink(pdf_path)


def render_document_list() -> None:
    """Render the list of processed documents."""
    if not st.session_state.processed_documents:
        st.info("No document loaded. Please upload a PDF file to start.")
        return

    st.header("📚 Processed Documents")

    doc_options = {st.session_state.filenames[h]: h for h in st.session_state.filenames}
    active_filename = st.session_state.filenames.get(st.session_state.active_document_hash)
    active_index = list(doc_options.keys()).index(active_filename) if active_filename else 0

    selected_filename = st.radio(
        "Select document to chat with:",
        options=list(doc_options.keys()),
        index=active_index,
        key="doc_selector",
    )

    selected_hash = doc_options[selected_filename]

    if st.session_state.active_document_hash != selected_hash:
        st.session_state.active_document_hash = selected_hash
        st.session_state.messages = []
        st.rerun()

    active_doc_data = st.session_state.processed_documents[st.session_state.active_document_hash]
    num_chunks = len(active_doc_data.get("chunk_texts", []))

    st.markdown(f"**Active:** {selected_filename}")
    st.markdown(f"**Total Chunks:** {num_chunks}")

    show_chunks = st.checkbox("Show document chunks preview", value=False)
    if show_chunks:
        for i, chunk in enumerate(active_doc_data.get("chunk_texts", [])[:5]):
            st.markdown(f"**Chunk {i + 1}**")
            st.text(chunk[:200] + "..." if len(chunk) > 200 else chunk)
        if num_chunks > 5:
            st.markdown(f"*...and {num_chunks - 5} more chunks*")


def render_model_selection() -> tuple[str | None, str | None]:
    """Render the model selection dropdown in the sidebar."""
    initialize_model_settings()
    from paperchat.llms.common import list_models
    from paperchat.utils.api_keys import get_api_key, get_available_providers

    all_models = []
    active_provider = st.session_state.config.get("provider_name", "")
    active_model = st.session_state.config.get("model_id", "")

    for provider in get_available_providers():
        if not get_api_key(provider):
            continue

        models = list_models(provider_name=provider)
        for model in models:
            model_display = f"{model['id']}"
            all_models.append(
                {"display": model_display, "provider": provider, "model_id": model["id"]}
            )

    if not all_models:
        return None, None

    selected_index = 0
    for i, model_info in enumerate(all_models):
        if model_info["provider"] == active_provider and model_info["model_id"] == active_model:
            selected_index = i
            break

    selected_model = st.selectbox(
        "Model:",
        options=range(len(all_models)),
        format_func=lambda i: all_models[i]["display"],
        index=selected_index,
    )

    selected_provider = all_models[selected_model]["provider"]
    selected_model_id = all_models[selected_model]["model_id"]

    if selected_provider != active_provider or selected_model_id != active_model:
        st.session_state.config.update(
            {
                "provider_name": selected_provider,
                "model_id": selected_model_id,
            }
        )

        temperature = st.session_state.config.get("temperature", 0.7)
        update_global_settings(selected_provider, selected_model_id, temperature)
        refresh_model_state()
        st.success(f"Switched to {selected_provider.capitalize()} model: {selected_model_id}")

    return selected_provider, selected_model_id


def render_advanced_settings(selected_provider: str, selected_model_id: str) -> None:
    """Render advanced settings like temperature and retrieval settings."""
    with st.expander("Advanced Settings", expanded=False):
        if selected_provider and selected_model_id:
            current_temp = st.session_state.config.get("temperature", 0.7)
            temperature = st.slider(
                "Temperature",
                min_value=0.0,
                max_value=2.0,
                value=current_temp,
                step=0.1,
                help="Controls randomness: 0=deterministic, 2=creative",
            )

            if temperature != current_temp:
                st.session_state.config["temperature"] = temperature
                update_global_settings(selected_provider, selected_model_id, temperature)
                refresh_model_state()

        st.markdown("**Retrieval settings:**")
        top_k = st.slider(
            "Chunks to retrieve",
            min_value=1,
            max_value=10,
            value=st.session_state.config.get("top_k", 5),
            help="Higher values retrieve more chunks but may include less relevant information",
        )

        if "config" in st.session_state:
            st.session_state.config["top_k"] = top_k


def render_sidebar() -> None:
    """Render the app sidebar."""
    st.header("📄 Document")

    with st.expander("Upload Document", expanded=True):
        render_upload_section()

    if st.session_state.processed_documents:
        with st.expander("Processed Documents", expanded=True):
            render_document_list()
    else:
        with st.expander("Current Document", expanded=False):
            st.info("No document loaded. Please upload a PDF file to start.")

    st.divider()

    st.header("⚙️ Settings")

    if st.button("Open Settings", key="open_settings"):
        st.session_state.show_settings = True
        st.rerun()

    selected_provider, selected_model_id = render_model_selection()
    render_advanced_settings(selected_provider, selected_model_id)

    st.divider()

    st.header("🔄 Actions")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("🗑️ Clear Chat"):
            st.session_state.messages = []
            st.rerun()

    with col2:
        if st.button("🔄 Reset All"):
            st.session_state.processed_documents = {}
            st.session_state.messages = []
            st.rerun()
