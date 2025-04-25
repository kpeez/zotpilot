"""
Document handling UI components for Streamlit.
"""

import hashlib
import logging
import os
import tempfile
from pathlib import Path

import streamlit as st

from paperchat.llms.common import list_llm_models
from paperchat.ui.common import refresh_model_state
from paperchat.ui.settings import (
    update_global_settings,
)
from paperchat.utils.api_keys import get_api_key, get_available_providers

logger = logging.getLogger("document_ui")


def render_compact_settings_ui() -> None:
    """Render a compact settings UI for the sidebar."""

    all_models = []
    active_provider = st.session_state.config.get("provider_name", "")
    active_model = st.session_state.config.get("model_id", "")

    for provider in get_available_providers():
        if not get_api_key(provider):
            continue

        models = list_llm_models(provider_name=provider)
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


def check_vector_store_for_document(content_hash: str, original_filename: str) -> bool:
    """Check if a document exists in the vector store.

    Args:
        content_hash: Hash of the document
        original_filename: Original filename

    Returns:
        True if document was found in vector store, False otherwise
    """
    vector_store = st.session_state.vector_store

    exists, doc_info = vector_store.get_document_info(content_hash)

    if exists:
        collection_name = doc_info["collection_name"]
        logger.debug(f"Found document in registry: {original_filename} -> {collection_name}")

        collections = vector_store.list_collections()

        if collection_name in collections:
            info = vector_store.get_collection_info(collection_name)

            if info["count"] > 0:
                doc_data = vector_store.get_document_texts(collection_name)

                st.session_state.processed_documents[content_hash] = {
                    "collection_name": collection_name,
                    "chunk_texts": doc_data["documents"],
                    "chunk_metadata": doc_data["metadatas"],
                }

                st.session_state.filenames[content_hash] = original_filename
                st.session_state.document_sources[content_hash] = "vector_store"

                return True

    return False


def _process_uploaded_pdf(file_content: bytes, original_filename: str, content_hash: str) -> None:
    """Helper function to process a new PDF file and update state."""
    with st.spinner(f"Processing {original_filename}..."):
        pdf_path = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(file_content)
                pdf_path = tmp_file.name

            original_stem = Path(original_filename).stem
            success = st.session_state.rag_pipeline.vector_store.add_document(
                pdf_path=pdf_path,
                source_id=original_stem,
            )

            if success:
                st.session_state.filenames[content_hash] = original_filename
                st.session_state.document_sources[content_hash] = "vector_store"
                st.session_state.messages = []

                if "selected_documents" in st.session_state and isinstance(
                    st.session_state.selected_documents, list
                ):
                    if original_stem not in st.session_state.selected_documents:
                        st.session_state.selected_documents.append(original_stem)
                        logger.info(f"Added '{original_stem}' to selected documents.")
                else:
                    st.session_state.selected_documents = [original_stem]
                    logger.info(f"Initialized selected documents with '{original_stem}'.")

                st.success(f"Document processed and added: {original_filename}")
                st.rerun()
            else:
                st.error(f"Failed to process {original_filename}. Check logs.")

        except Exception as e:
            logger.exception(f"Error processing uploaded file {original_filename}: {e}")
            st.error(f"Error processing document: {e}")
        finally:
            if pdf_path and os.path.exists(pdf_path):
                try:
                    os.unlink(pdf_path)
                except Exception as e:
                    logger.error(f"Failed to delete temp file {pdf_path}: {e}")


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

        # check if already known
        if content_hash in st.session_state.filenames:
            st.success(f"Document ready: {original_filename}")
            # We might re-introduce setting active_document_hash if needed for single-doc focus later
            # if st.session_state.active_document_hash != content_hash:
            #     st.session_state.active_document_hash = content_hash
            #     st.session_state.messages = []
            #     st.rerun()
        else:
            _process_uploaded_pdf(file_content, original_filename, content_hash)


def render_document_list() -> None:
    """Render the list of processed documents."""
    if not st.session_state.filenames:
        st.info("No document loaded. Please upload a PDF file to start.")
        return

    st.header("📚 Loaded Documents")

    doc_items = [
        (st.session_state.filenames[hash_val], hash_val) for hash_val in st.session_state.filenames
    ]

    if not doc_items:
        st.info("Upload a PDF to begin.")
        return

    doc_items.sort()
    cols = st.columns([5, 1])

    with cols[0]:
        st.markdown("**Select a document to chat with:**")

    active_hash = st.session_state.active_document_hash

    selected_doc_hash = None

    for filename, file_hash in doc_items:
        source_icon = "💾"
        tooltip = f"Source: {filename} (Stored)"

        label = f"{source_icon} {filename}"

        button_type = "primary" if file_hash == active_hash else "secondary"
        button_key = f"doc_select_{file_hash}"

        if st.button(
            label, key=button_key, help=tooltip, type=button_type, use_container_width=True
        ):
            selected_doc_hash = file_hash

    if selected_doc_hash and selected_doc_hash != active_hash:
        st.session_state.active_document_hash = selected_doc_hash
        st.session_state.messages = []
        st.rerun()


def render_model_selection() -> tuple[str | None, str | None]:
    """Render the model selection dropdown in the sidebar."""
    all_models = []
    active_provider = st.session_state.config.get("provider_name", "")
    active_model = st.session_state.config.get("model_id", "")

    for provider in get_available_providers():
        if not get_api_key(provider):
            continue

        models = list_llm_models(provider_name=provider)
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


def _render_document_selector(doc_names: list[str]) -> None:
    """Renders the document selection checkboxes and syncs state."""
    # Apply custom CSS to shrink checkbox label font size
    st.markdown(
        """
    <style>
    div[data-testid="stCheckbox"] label p {
        font-size: 0.9em !important; /* Adjust size as needed */
    }
    /* Add some margin below buttons */
    div[data-testid="stHorizontalBlock"] button {
        margin-bottom: 5px;
    }
    </style>
    """,
        unsafe_allow_html=True,
    )

    # default to all documents
    if "selected_documents" not in st.session_state:
        st.session_state.selected_documents = doc_names[:]

    with st.expander("Select Chat Context", expanded=True):
        st.caption("Check/uncheck boxes to select documents. Default is all selected.")

        col1, col2 = st.columns(2)
        select_all_clicked = col1.button("Select All", key="select_all_docs")
        deselect_all_clicked = col2.button("Deselect All", key="deselect_all_docs")

        if select_all_clicked and set(st.session_state.selected_documents) != set(doc_names):
            st.session_state.selected_documents = doc_names[:]
            logger.info("Selected all documents.")
            st.rerun()

        if deselect_all_clicked and st.session_state.selected_documents:
            st.session_state.selected_documents = []
            logger.info("Deselected all documents.")
            st.rerun()

        # render checkboxes based on current session state
        if doc_names:
            current_ui_selection = []
            for doc_name in doc_names:
                is_selected = doc_name in st.session_state.selected_documents
                if st.checkbox(doc_name, value=is_selected, key=f"cb_{doc_name}"):
                    current_ui_selection.append(doc_name)

            # compare UI state with session state and update if checkbox interaction caused change
            # this check is important to only trigger rerun on direct checkbox clicks,
            # not after button clicks (which already reran).
            if set(current_ui_selection) != set(st.session_state.selected_documents):
                st.session_state.selected_documents = current_ui_selection
                logger.info(
                    f"Updated selected documents via checkboxes: {st.session_state.selected_documents}"
                )
                st.rerun()
        else:
            st.info("No documents found in the vector store.")
            if st.session_state.selected_documents:
                st.session_state.selected_documents = []
                # no rerun needed here as it implies doc_names just became empty


def render_sidebar() -> None:
    """Render the app sidebar."""
    st.header("📚 Sources")

    with st.expander("Add Document", expanded=True):
        render_upload_section()

    try:
        if "rag_pipeline" in st.session_state and st.session_state.rag_pipeline:
            vector_store = st.session_state.rag_pipeline.vector_store
            doc_list = vector_store.list_documents()
            doc_names = sorted([doc.get("source", "Unknown") for doc in doc_list])

            _render_document_selector(doc_names)

        else:
            st.warning("RAG Pipeline not initialized. Cannot list documents.")
            if "selected_documents" not in st.session_state or st.session_state.selected_documents:
                st.session_state.selected_documents = []

    except AttributeError as e:
        logger.error(f"Attribute error accessing vector store: {e}")
        st.error("Could not access vector store components.")
    except Exception as e:
        logger.error(f"Failed to list documents from vector store: {e}")
        st.error("Could not retrieve document list.")

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
