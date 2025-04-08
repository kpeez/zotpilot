"""
Document handling UI components for Streamlit.
"""

import hashlib
import os
import tempfile

import streamlit as st

from paperchat.ingestion import process_document
from paperchat.ui.common import show_info, show_warning
from paperchat.ui.settings import (
    initialize_model_settings,
    update_global_settings,
)


def render_compact_settings_ui() -> None:
    """Render a compact settings UI for the sidebar."""
    initialize_model_settings()

    selected_provider = st.session_state.active_provider

    providers = list(st.session_state.provider_settings)
    if providers:
        col1, col2 = st.columns([1, 2])
        with col1:
            st.markdown("**Provider:**")
        with col2:
            provider_index = (
                providers.index(selected_provider) if selected_provider in providers else 0
            )
            selected_provider = st.radio(
                "",
                options=providers,
                format_func=lambda x: x.capitalize(),
                index=provider_index,
                key="sidebar_provider_selector",
                horizontal=True,
                label_visibility="collapsed",
            )
    else:
        show_warning("No providers configured. Please add API keys in settings.")
        return

    current_settings = st.session_state.provider_settings.get(selected_provider, {})

    col1, col2 = st.columns([1, 2])
    with col1:
        st.markdown("**Model:**")
    with col2:
        all_models = (
            ["gpt-4o", "gpt-4o-mini"]
            if selected_provider == "openai"
            else ["claude-3.7-sonnet", "claude-3.5-sonnet"]
        )
        selected_model = st.selectbox(
            "",
            options=all_models,
            index=all_models.index(current_settings.get("model", all_models[0]))
            if current_settings.get("model") in all_models
            else 0,
            label_visibility="collapsed",
        )

    if selected_provider in st.session_state.provider_settings:
        st.session_state.provider_settings[selected_provider]["model"] = selected_model

    st.markdown("---")
    st.markdown("**Retrieval settings:**")
    top_k = st.slider(
        "Chunks to retrieve",
        min_value=1,
        max_value=10,
        value=st.session_state.settings.get("top_k", 5),
        help="Higher values retrieve more chunks but may include less relevant information",
    )

    temperature = st.session_state.provider_settings.get(selected_provider, {}).get(
        "temperature", 0.7
    )
    update_global_settings(selected_provider, selected_model, temperature)

    if "settings" in st.session_state:
        st.session_state.settings.update({"top_k": top_k})


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
        show_info("No document loaded. Please upload a PDF file to start.")
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
            show_info("No document loaded. Please upload a PDF file to start.")

    st.divider()

    st.header("⚙️ Settings")

    if st.button("Open Settings", key="open_settings"):
        st.session_state.show_settings = True
        st.rerun()

    with st.expander("Model & Retrieval", expanded=False):
        render_compact_settings_ui()

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
