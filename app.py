"""
app.py
------
Streamlit front-end for the Finance & Marketing Interview Assistant.

This file only handles the UI: layout, chat history, and displaying
results. All the actual RAG logic lives in rag.py so this file stays
easy to read.
"""
import os
import streamlit as st

from rag import answer_question, load_vectorstore
from utils import get_openrouter_api_key, get_pdf_files, load_env

SUGGESTED_QUESTIONS = [
    "Explain Porter's Five Forces",
    "What is SWOT Analysis?",
    "Difference between NPV and IRR",
    "Explain CAPM",
    "What is STP Marketing?",
    "What is EBITDA?",
    "Explain Working Capital",
    "Tell me about the STAR interview framework",
]


def init_session_state() -> None:
    """Set up chat history the first time the app runs."""
    if "messages" not in st.session_state:
        st.session_state.messages = []  # list of {"role", "content", "sources"}


def render_sidebar() -> None:
    """Sidebar with suggested questions and a button to clear the chat."""
    with st.sidebar:
        st.header("Suggested Questions")
        for question in SUGGESTED_QUESTIONS:
            if st.button(question, use_container_width=True):
                st.session_state.pending_question = question

        st.divider()
        if st.button("Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()


def render_chat_history() -> None:
    """Re-display all previous messages, including sources for answers."""
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("sources"):
                render_sources(message["sources"])


def render_sources(sources: list) -> None:
    """Show the Sources section under an answer."""
    st.markdown("**Sources:**")
    for source in sources:
        st.markdown(f"- {source.filename} (Page {source.page})")


def render_retrieved_context(chunks: list) -> None:
    """Collapsed-by-default expander showing the raw chunks used, for debugging."""
    with st.expander(f"Retrieved Context ({len(chunks)} chunk(s))", expanded=False):
        for i, chunk in enumerate(chunks, start=1):
            st.markdown(f"**Chunk {i}**")
            st.text(chunk.page_content)


def handle_question(question: str, vectorstore) -> None:
    """Run the RAG pipeline for a question and render the result in the chat."""
    st.session_state.messages.append({"role": "user", "content": question, "sources": None})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching interview material and generating an answer..."):
            try:
                result = answer_question(vectorstore, question)
            except ValueError as exc:
                st.warning(str(exc))
                return
            except RuntimeError as exc:
                st.error(str(exc))
                return

        st.markdown(result.answer)
        if result.sources:
            render_sources(result.sources)
        if result.retrieved_chunks:
            render_retrieved_context(result.retrieved_chunks)

    st.session_state.messages.append(
        {"role": "assistant", "content": result.answer, "sources": result.sources}
    )


def main() -> None:
    st.set_page_config(page_title="Finance & Marketing Interview Assistant", page_icon="💼")
    load_env()
    if "OPENROUTER_API_KEY" in st.secrets:
        os.environ["OPENROUTER_API_KEY"] = st.secrets["OPENROUTER_API_KEY"]

    if "OPENROUTER_MODEL" in st.secrets:
        os.environ["OPENROUTER_MODEL"] = st.secrets["OPENROUTER_MODEL"]

    init_session_state()

    st.title("Finance & Marketing Interview Assistant")
    st.caption("Prepare smarter using your interview material.")

    render_sidebar()

    # Guard rail: check the API key early so the error is clear, not a
    # stack trace when the user finally sends a question.
    try:
        get_openrouter_api_key()
    except ValueError as exc:
        st.error(str(exc))
        return

    # Guard rail: make sure there is something to search before showing the chat
    # if not get_pdf_files():
    #     st.warning(
    #         "No PDFs found in the documents/ folder. Add your interview prep "
    #         "PDFs there and run `python ingest.py` before chatting."
    #     )
    #     return

    try:
        vectorstore = load_vectorstore()
    except Exception as exc:
        st.error(f"Could not load the document database: {exc}")
        return

    if vectorstore._collection.count() == 0:
        st.warning(
            "Your document database is empty. Run `python ingest.py` to "
            "process the PDFs in documents/ before asking questions."
        )
        return

    render_chat_history()

    # A sidebar button sets this; treat it just like typed input
    question = st.chat_input("Ask a finance or marketing interview question...")
    if "pending_question" in st.session_state:
        question = st.session_state.pop("pending_question")

    if question:
        handle_question(question, vectorstore)


if __name__ == "__main__":
    main()
