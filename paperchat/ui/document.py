"""
Document handling UI components for Streamlit.
"""

import hashlib
import os
import tempfile

import streamlit as st

from paperchat.ingestion import process_document
from paperchat.ui.common import show_info
from paperchat.ui.settings import PROVIDER_MODELS


def render_upload_section():
    """Render document upload section in the sidebar."""
    st.markdown("Upload a PDF to start chatting with its content.")

    uploaded_file = st.file_uploader(
        "Choose a PDF file", type=["pdf"], help="Select a PDF file to upload and process"
    )

    if uploaded_file is not None:
        original_filename = uploaded_file.name
        file_content = uploaded_file.getvalue()
        content_hash = hashlib.md5(file_content).hexdigest()

        # check if document is already processed
        if content_hash in st.session_state.processed_documents:
            st.success(f"Document already processed: {original_filename}")
            # if it's not the active one, make it active and clear chat
            if st.session_state.active_document_hash != content_hash:
                st.session_state.active_document_hash = content_hash
                st.session_state.messages = []
                st.rerun()  # rerun to reflect the change in active doc and clear chat
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


def render_document_list():
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


def render_sidebar():
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

    # Add Settings button
    if st.button("Open Settings", key="open_settings"):
        st.session_state.show_settings = True
        st.rerun()

    with st.expander("Adjust model and retrieval settings", expanded=False):
        model = st.selectbox(
            "Model",
            options=PROVIDER_MODELS.get("openai", ["gpt-4o"]),
            index=0,
            help="Select the language model to use",
        )

        st.subheader("Retrieval Settings")
        top_k = st.slider(
            "Number of chunks to retrieve",
            min_value=1,
            max_value=10,
            value=st.session_state.settings["top_k"],
            help="Higher values retrieve more document chunks but may include less relevant information",
        )

        st.subheader("LLM Settings")
        temperature = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=2.0,
            value=st.session_state.settings["temperature"],
            step=0.1,
            help="Higher values make output more creative, lower values more deterministic",
        )

        st.session_state.settings.update(
            {
                "top_k": top_k,
                "temperature": temperature,
                "model": model,
            }
        )

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
