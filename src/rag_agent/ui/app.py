"""
app.py
======
Streamlit user interface for the Deep Learning RAG Interview Prep Agent.

Workshop version (Markdown-only RAG demo):
  - Left sidebar: document ingestion + corpus stats
  - Centre: document viewer (simple placeholder)
  - Right: RAG question-and-answer chat with source citations

Flow for each question:
  query -> VectorStoreManager.query() (ChromaDB retrieval)
        -> Groq LLM answers using ONLY the retrieved context
        -> answer + sources shown in the chat

PEP 8 | OOP | Single Responsibility
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import streamlit as st
from langchain_core.messages import HumanMessage, SystemMessage

from rag_agent.config import LLMFactory, get_settings
from rag_agent.corpus.chunker import DocumentChunker
from rag_agent.vectorstore.store import VectorStoreManager


# ---------------------------------------------------------------------------
# Cached Resources
# ---------------------------------------------------------------------------
# st.cache_resource keeps these objects alive across Streamlit reruns, so
# ChromaDB, the embedding model and the LLM client are created only once.


@st.cache_resource
def get_vector_store() -> VectorStoreManager:
    """Return the singleton VectorStoreManager."""
    return VectorStoreManager()


@st.cache_resource
def get_chunker() -> DocumentChunker:
    """Return the singleton DocumentChunker."""
    return DocumentChunker()


@st.cache_resource
def get_llm():
    """Return the cached chat model (Groq, configured in .env)."""
    return LLMFactory().create()


# ---------------------------------------------------------------------------
# Session State Initialisation
# ---------------------------------------------------------------------------


def initialise_session_state() -> None:
    """Initialise all st.session_state keys on first run."""
    defaults = {
        "chat_history": [],  # list of {"role", "content", "sources", "no_context_found"}
        "last_ingestion_result": None,
    }
    for key, default in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default


# ---------------------------------------------------------------------------
# Ingestion Panel (Sidebar)
# ---------------------------------------------------------------------------


def render_ingestion_panel(
    store: VectorStoreManager,
    chunker: DocumentChunker,
) -> None:
    """Upload Markdown files, chunk them and store them in ChromaDB."""
    st.sidebar.header("📂 Corpus Ingestion")

    uploaded_files = st.sidebar.file_uploader(
        "Upload Markdown notes",
        type=["md"],
        accept_multiple_files=True,
    )

    if st.sidebar.button("Ingest Documents", disabled=not uploaded_files):
        total_added = 0
        total_skipped = 0
        errors: list[str] = []

        with st.sidebar.status("Ingesting...", expanded=False):
            with tempfile.TemporaryDirectory() as tmp_dir:
                for uploaded in uploaded_files:
                    # Keep the original filename so metadata (topic, source)
                    # is inferred from it, e.g. lstm_intermediate.md
                    file_path = Path(tmp_dir) / uploaded.name
                    file_path.write_bytes(uploaded.getvalue())
                    try:
                        chunks = chunker.chunk_file(file_path)
                        result = store.ingest(chunks)
                        total_added += result.ingested
                        total_skipped += result.skipped
                        errors.extend(result.errors)
                    except Exception as exc:  # show the error, keep going
                        errors.append(f"{uploaded.name}: {exc}")

        st.session_state.last_ingestion_result = {
            "added": total_added,
            "skipped": total_skipped,
            "errors": errors,
        }

    result = st.session_state.last_ingestion_result
    if result:
        message = (
            f"{result['added']} chunks added, "
            f"{result['skipped']} duplicates skipped"
        )
        if result["errors"]:
            st.sidebar.error(message + "\n\n" + "\n".join(result["errors"]))
        elif result["added"] > 0:
            st.sidebar.success(message)
        else:
            st.sidebar.warning(message)

    st.sidebar.info("Upload .md files to populate the corpus.")


def render_corpus_stats(store: VectorStoreManager) -> None:
    """Show total chunk count and topics in the sidebar."""
    stats = store.get_collection_stats()
    st.sidebar.metric("ChromaDB chunks", stats["total_chunks"])
    if stats["topics"]:
        st.sidebar.write("Topics:", ", ".join(stats["topics"]))
    if stats["sources"]:
        st.sidebar.write("Sources:", ", ".join(stats["sources"]))


# ---------------------------------------------------------------------------
# Document Viewer Panel (Centre)
# ---------------------------------------------------------------------------


def render_document_viewer(store: VectorStoreManager) -> None:
    """Simple summary of what is in the corpus."""
    st.subheader("📄 Corpus")
    stats = store.get_collection_stats()
    if stats["total_chunks"] == 0:
        st.info("Ingest documents using the sidebar to view content here.")
        return
    st.write(f"**{stats['total_chunks']} chunks** stored in ChromaDB.")
    for source in stats["sources"]:
        st.markdown(f"- `{source}`")


# ---------------------------------------------------------------------------
# Chat Interface Panel (Right)
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are a helpful deep learning tutor. Answer the question using ONLY "
    "the context provided. If the context does not contain the answer, say "
    "you could not find it in the uploaded notes. Keep answers clear and concise."
)


def answer_question(query: str, store: VectorStoreManager, llm) -> dict:
    """Retrieve relevant chunks and ask the LLM to answer from them."""
    chunks = store.query(query)

    # Hallucination guard: no relevant context -> do not call the LLM
    if not chunks:
        return {
            "role": "assistant",
            "content": "I couldn't find anything relevant to that in the uploaded notes.",
            "sources": [],
            "no_context_found": True,
        }

    context = "\n\n".join(
        f"Source {i + 1} {chunk.to_citation()}:\n{chunk.chunk_text}"
        for i, chunk in enumerate(chunks)
    )
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=f"Context:\n{context}\n\nQuestion: {query}"),
    ]
    response = llm.invoke(messages)

    sources = [f"{chunk.to_citation()}  (score: {chunk.score:.2f})" for chunk in chunks]
    return {
        "role": "assistant",
        "content": response.content,
        "sources": sources,
        "no_context_found": False,
    }


def render_chat_interface(store: VectorStoreManager, llm) -> None:
    """RAG question-and-answer chat with source citations."""
    st.subheader("💬 RAG Chat")

    chat_container = st.container(height=450)
    with chat_container:
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                if message.get("sources"):
                    with st.expander("📎 Sources", expanded=True):
                        for source in message["sources"]:
                            st.caption(source)
                if message.get("no_context_found"):
                    st.warning("⚠️ No relevant content found in corpus.")

    query = st.chat_input("Ask a question about your notes...")
    if query:
        st.session_state.chat_history.append({"role": "user", "content": query})
        with st.spinner("Retrieving context and generating answer..."):
            try:
                reply = answer_question(query, store, llm)
            except Exception as exc:
                reply = {
                    "role": "assistant",
                    "content": f"Error: {exc}",
                    "sources": [],
                    "no_context_found": False,
                }
        st.session_state.chat_history.append(reply)
        st.rerun()


# ---------------------------------------------------------------------------
# Main Application
# ---------------------------------------------------------------------------


def main() -> None:
    """
    Application entry point.

    Run with: uv run streamlit run src/rag_agent/ui/app.py
    """
    settings = get_settings()

    st.set_page_config(
        page_title=settings.app_title,
        page_icon="🧠",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.title(f"🧠 {settings.app_title}")
    st.caption("RAG demo: Markdown notes → ChromaDB → Groq LLM answers with sources")

    initialise_session_state()

    store = get_vector_store()
    chunker = get_chunker()
    llm = get_llm()

    # Sidebar
    render_ingestion_panel(store, chunker)
    render_corpus_stats(store)

    # Main content area — two columns
    viewer_col, chat_col = st.columns([1, 2], gap="large")
    with viewer_col:
        render_document_viewer(store)
    with chat_col:
        render_chat_interface(store, llm)


if __name__ == "__main__":
    main()