import hashlib
import os
import tempfile
from pathlib import Path

import streamlit as st

from zotpilot.embeddings import EmbeddingModel
from zotpilot.ingestion import process_document
from zotpilot.llm import get_openai_client, rag_pipeline
from zotpilot.utils.formatting import (
    format_response_with_citations,
    format_retrieved_chunks_for_display,
)
from zotpilot.utils.settings import DEFAULT_MAX_TOKENS, DEFAULT_MODEL, DEFAULT_TEMPERATURE

os.environ["TOKENIZERS_PARALLELISM"] = "false"

st.set_page_config(
    page_title="ZotPilot - Academic PDF Chat",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
    .document-card {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 10px;
    }
    .stButton button {
        width: 100%;
    }
    .action-buttons {
        display: flex;
        flex-direction: column;
        gap: 10px;
    }
</style>
""",
    unsafe_allow_html=True,
)


def initialize_session():
    """Initialize all session state variables and models in one place."""
    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "document_data" not in st.session_state:
        st.session_state.document_data = None

    if "settings" not in st.session_state:
        st.session_state.settings = {
            "top_k": 5,
            "temperature": DEFAULT_TEMPERATURE,
            "max_tokens": DEFAULT_MAX_TOKENS,
            "model": DEFAULT_MODEL,
        }

    if "embedding_model" not in st.session_state:
        st.session_state.embedding_model = EmbeddingModel()

    if "llm_client" not in st.session_state:
        st.session_state.llm_client = get_openai_client()


st.title("🤖 ZotPilot - Chat with your research library")
initialize_session()


with st.sidebar:
    st.header("📄 Document")

    st.markdown("Upload a PDF to start chatting with its content.")

    uploaded_file = st.file_uploader(
        "Choose a PDF file", type=["pdf"], help="Select a PDF file to upload and process"
    )

    if uploaded_file is not None:
        current_doc_name = (
            st.session_state.document_data.get("collection_name", None)
            if st.session_state.document_data
            else None
        )

        if current_doc_name != Path(uploaded_file.name).stem:
            file_content = uploaded_file.getvalue()
            content_hash = hashlib.md5(file_content).hexdigest()
            if (
                hasattr(st.session_state, "last_processed_hash")
                and st.session_state.last_processed_hash == content_hash
            ):
                st.success(f"Document already processed: {uploaded_file.name}")
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

                    st.session_state.document_data = document_data
                    st.session_state.last_processed_hash = content_hash
                    os.unlink(pdf_path)

                    progress_bar.progress(100, "Complete!")
                    st.success(f"Document processed: {uploaded_file.name}")
                    st.session_state.messages = []
                except Exception as e:
                    progress_bar.progress(100, "Error!")
                    st.error(f"Error processing document: {e}")
                    if "pdf_path" in locals():
                        os.unlink(pdf_path)

    st.divider()

    if st.session_state.document_data is not None:
        st.markdown(
            """
        <div class="document-card">
            <h3>📝 Current Document</h3>
        """,
            unsafe_allow_html=True,
        )

        doc_name = st.session_state.document_data.get("collection_name", "Unknown")
        num_chunks = len(st.session_state.document_data.get("chunk_texts", []))

        st.markdown(f"**Title:** {doc_name}")
        st.markdown(f"**Total Chunks:** {num_chunks}")

        with st.expander("Preview document chunks"):
            for i, chunk in enumerate(st.session_state.document_data.get("chunk_texts", [])[:5]):
                st.markdown(f"**Chunk {i + 1}**")
                st.text(chunk[:200] + "..." if len(chunk) > 200 else chunk)

            if num_chunks > 5:
                st.markdown(f"*...and {num_chunks - 5} more chunks*")

        st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.info("No document loaded. Please upload a PDF file to start.")

    st.divider()

    st.header("⚙️ Settings")

    with st.expander("Adjust model and retrieval settings", expanded=False):
        model = st.selectbox(
            "Model", options=[DEFAULT_MODEL], index=0, help="Select the language model to use"
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
            max_value=1.0,
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
            st.session_state.document_data = None
            st.session_state.messages = []
            st.rerun()

if st.session_state.document_data is None:
    st.markdown("""
    ### 👋 Welcome to ZotPilot!

    To get started:
    1. Upload a PDF using the sidebar
    2. Wait for the document to be processed
    3. Ask questions about the document content

    ZotPilot will use AI to retrieve relevant information and provide answers based on the document.
    """)
else:
    chat_container = st.container()

    with chat_container:
        if st.session_state.messages:
            for msg in st.session_state.messages:
                with st.chat_message(msg["role"]):
                    if msg["role"] == "assistant":
                        st.markdown(msg["content"], unsafe_allow_html=True)

                        if msg.get("sources"):
                            with st.expander("📚 Sources", expanded=False):
                                sources_md = format_retrieved_chunks_for_display(msg["sources"])
                                st.markdown(sources_md, unsafe_allow_html=True)
                    else:
                        st.markdown(msg["content"])
        else:
            st.info("Chat is empty. Ask a question about the document to begin.")

    user_query = st.chat_input("Ask a question about the document")

    if user_query:
        st.session_state.messages.append({"role": "user", "content": user_query})

        with st.chat_message("user"):
            st.markdown(user_query)

        response_placeholder = st.empty()

        with st.spinner("Generating response..."):
            try:
                document_data = st.session_state.document_data
                settings = st.session_state.settings

                response, retrieved_chunks = rag_pipeline(
                    query=user_query,
                    document_data=st.session_state.document_data,
                    top_k=settings["top_k"],
                    model=settings["model"],
                    temperature=settings["temperature"],
                    max_tokens=settings["max_tokens"],
                    stream=False,
                    embedding_model=st.session_state.embedding_model,
                    client=st.session_state.llm_client,
                )

                formatted_response = format_response_with_citations(response)

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": formatted_response,
                        "raw_content": response,
                        "sources": retrieved_chunks,
                    }
                )

                with response_placeholder.chat_message("assistant"):
                    st.markdown(formatted_response, unsafe_allow_html=True)

                    if retrieved_chunks:
                        with st.expander("📚 View Sources for this response", expanded=False):
                            sources_md = format_retrieved_chunks_for_display(retrieved_chunks)
                            st.markdown(sources_md, unsafe_allow_html=True)

            except Exception as e:
                error_message = f"Error: {e!s}"
                st.error(error_message)

                st.session_state.messages.append(
                    {"role": "assistant", "content": error_message, "sources": []}
                )

                with response_placeholder.chat_message("assistant"):
                    st.error(error_message)
