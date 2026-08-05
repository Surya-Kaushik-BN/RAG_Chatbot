"""
rag.py
------
Core RAG (Retrieval-Augmented Generation) logic:
1. Retrieve the most relevant chunks from ChromaDB for a user question.
2. Send those chunks + the question to the OpenRouter LLM.
3. Return the answer along with the sources used.

This module has no Streamlit code in it, so it could be reused or tested
independently of the UI.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from openai import OpenAI

from prompts import SYSTEM_PROMPT, build_user_prompt
from utils import (
    CHROMA_DIR,
    COLLECTION_NAME,
    get_openrouter_api_key,
    get_openrouter_model,
)

TOP_K = 4  # number of chunks to retrieve per question

# OpenRouter exposes an OpenAI-compatible API, so we can reuse the openai client
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


@dataclass
class Source:
    """A single source reference shown under an answer."""
    filename: str
    page: int


@dataclass
class RagAnswer:
    """The full result of answering a question: text + sources + raw chunks."""
    answer: str
    sources: list[Source] = field(default_factory=list)
    retrieved_chunks: list[Document] = field(default_factory=list)


def load_vectorstore() -> Chroma:
    """Connect to the existing ChromaDB collection created by ingest.py."""
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(CHROMA_DIR),
    )


def retrieve_chunks(vectorstore: Chroma, question: str, top_k: int = TOP_K) -> list[Document]:
    """Run a similarity search and return the top-k most relevant chunks."""
    return vectorstore.similarity_search(question, k=top_k)


def format_context(chunks: list[Document]) -> str:
    """Combine retrieved chunks into a single text block for the LLM prompt."""
    parts = []
    for i, chunk in enumerate(chunks, start=1):
        filename = Path(chunk.metadata.get("source", "unknown")).name.split('\\')[-1]  # Show only the file name, not the full path
        page = chunk.metadata.get("page", 0) + 1  # PyPDFLoader pages are 0-indexed
        parts.append(f"[Source {i}: {filename}, Page {page}]\n{chunk.page_content}")
    return "\n\n".join(parts)


def extract_sources(chunks: list[Document]) -> list[Source]:
    """Turn chunk metadata into a de-duplicated list of Source objects."""
    seen = set()
    sources = []
    for chunk in chunks:
        filename = Path(chunk.metadata.get("source", "unknown")).name
        page = chunk.metadata.get("page", 0) + 1
        key = (filename, page)
        filename_display = filename.split('\\')[-1]  # Show only the file name, not the full path
        if key not in seen:
            seen.add(key)
            sources.append(Source(filename=filename_display, page=page))
    return sources


def call_llm(context: str, question: str) -> str:
    """Send the system prompt + context + question to the OpenRouter LLM."""
    api_key = get_openrouter_api_key()
    model = get_openrouter_model()

    client = OpenAI(base_url=OPENROUTER_BASE_URL, api_key=api_key)

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(context, question)},
        ],
        temperature=0.2,  # low temperature keeps answers grounded and consistent
    )
    return response.choices[0].message.content.strip()


def answer_question(vectorstore: Chroma, question: str) -> RagAnswer:
    """
    Full RAG pipeline for one question: retrieve -> build context -> call LLM.

    Raises ValueError for an empty question and RuntimeError if the LLM
    call fails, so the Streamlit layer can catch these and show a friendly
    message.
    """
    question = question.strip()
    if not question:
        raise ValueError("Please enter a question.")

    chunks = retrieve_chunks(vectorstore, question)

    if not chunks:
        return RagAnswer(
            answer="I couldn't find this information in the interview preparation material.",
            sources=[],
            retrieved_chunks=[],
        )

    context = format_context(chunks)

    try:
        answer_text = call_llm(context, question)
    except Exception as exc:
        raise RuntimeError(f"The AI model could not be reached: {exc}") from exc

    return RagAnswer(
        answer=answer_text,
        sources=extract_sources(chunks),
        retrieved_chunks=chunks,
    )
