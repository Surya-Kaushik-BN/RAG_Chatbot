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

from prompts import (
    SYSTEM_PROMPT,
    build_user_prompt,
    build_cheatsheet_generation_prompt,
    build_cheatsheet_subtopic_prompt,
    build_cheatsheet_topic_prompt,
    build_flashcard_generation_prompt,
    build_quiz_evaluation_prompt,
    build_quiz_generation_prompt,
)
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


def retrieve_chunks(
    vectorstore: Chroma,
    question: str,
    domain: str = "all",
    top_k: int = TOP_K,
) -> list[Document]:
    """Run a similarity search and return the top-k most relevant chunks."""
    metadata_filter = None
    if domain and domain.lower() != "all":
        metadata_filter = {"domain": domain.lower()}
    return vectorstore.similarity_search(question, k=top_k, filter=metadata_filter)


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


def call_llm(context: str, prompt: str, conversation: str | None = None) -> str:
    """Send the system prompt + context + prompt (+ optional conversation)
    to the OpenRouter LLM.
    """
    api_key = get_openrouter_api_key()
    model = get_openrouter_model()

    client = OpenAI(base_url=OPENROUTER_BASE_URL, api_key=api_key)

    user_prompt = build_user_prompt(context, prompt, conversation or "")

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,  # low temperature keeps answers grounded and consistent
    )
    return response.choices[0].message.content.strip()


def answer_question(
    vectorstore: Chroma,
    question: str,
    domain: str = "all",
    conversation: list[dict] | None = None,
) -> RagAnswer:
    """
    Full RAG pipeline for one question: retrieve -> build context -> call LLM.

    Raises ValueError for an empty question and RuntimeError if the LLM
    call fails, so the Streamlit layer can catch these and show a friendly
    message.
    """
    question = question.strip()
    if not question:
        raise ValueError("Please enter a question.")

    # Retrieve relevant context from the vectorstore
    chunks = retrieve_chunks(vectorstore, question, domain=domain)

    if not chunks:
        return RagAnswer(
            answer="I couldn't find this information in the interview preparation material.",
            sources=[],
            retrieved_chunks=[],
        )

    context = format_context(chunks)

    # Build a short conversation string from the provided conversation list
    conv_text = ""
    if conversation:
        # Keep only the last few turns (3) to avoid overly long prompts
        recent = conversation[-3:]
        parts = []
        for msg in recent:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            parts.append(f"{role.title()}: {content}")
        conv_text = "\n".join(parts)

    try:
        answer_text = call_llm(context, question, conversation=conv_text)
    except Exception as exc:
        raise RuntimeError(f"The AI model could not be reached: {exc}") from exc

    return RagAnswer(
        answer=answer_text,
        sources=extract_sources(chunks),
        retrieved_chunks=chunks,
    )


def generate_rapid_fire_question(
    vectorstore: Chroma,
    topic: str,
    difficulty: str,
    previous_questions: list[str],
    domain: str = "all",
    top_k: int = TOP_K * 3,
) -> RagAnswer:
    """Generate one new rapid-fire question plus its ideal answer and rubric."""
    prompt = build_quiz_generation_prompt(
        topic=topic,
        difficulty=difficulty,
        previous_questions="\n".join(previous_questions),
    )
    return answer_task(vectorstore, prompt, domain=domain, top_k=top_k)


def generate_flashcard(
    vectorstore: Chroma,
    topic: str,
    difficulty: str,
    previous_terms: list[str],
    domain: str = "all",
    top_k: int = TOP_K * 3,
) -> RagAnswer:
    """Generate exactly one flashcard from the selected domain and topic."""
    prompt = build_flashcard_generation_prompt(
        topic=topic,
        difficulty=difficulty,
        previous_terms="\n".join(previous_terms),
    )
    return answer_task(vectorstore, prompt, domain=domain, top_k=top_k)


def discover_cheatsheet_topics(
    vectorstore: Chroma,
    domain: str = "all",
    top_k: int = TOP_K * 3,
) -> RagAnswer:
    """Discover actual topic names from the selected domain documents."""
    prompt = build_cheatsheet_topic_prompt(domain=domain)
    return answer_task(vectorstore, prompt, domain=domain, top_k=top_k)


def discover_cheatsheet_subtopics(
    vectorstore: Chroma,
    topic: str,
    domain: str = "all",
    top_k: int = TOP_K * 3,
) -> RagAnswer:
    """Discover actual subtopics for a topic from the selected domain documents."""
    prompt = build_cheatsheet_subtopic_prompt(domain=domain, topic=topic)
    return answer_task(vectorstore, prompt, domain=domain, top_k=top_k)


def generate_cheatsheet(
    vectorstore: Chroma,
    domain: str,
    topic: str,
    subtopic: str,
    top_k: int = TOP_K * 3,
) -> RagAnswer:
    """Generate a focused cheatsheet for the selected topic/subtopic."""
    prompt = build_cheatsheet_generation_prompt(
        domain=domain,
        topic=topic,
        subtopic=subtopic,
    )
    return answer_task(vectorstore, prompt, domain=domain, top_k=top_k)


def evaluate_rapid_fire_answer(
    vectorstore: Chroma,
    question: str,
    ideal_answer: str,
    student_answer: str,
    domain: str = "all",
    top_k: int = TOP_K * 3,
) -> RagAnswer:
    """Evaluate a student's answer to the current rapid-fire question."""
    prompt = build_quiz_evaluation_prompt(
        question=question,
        ideal_answer=ideal_answer,
        student_answer=student_answer,
    )
    return answer_task(vectorstore, prompt, domain=domain, top_k=top_k)


def answer_task(
    vectorstore: Chroma,
    task_text: str,
    domain: str = "all",
    top_k: int = TOP_K * 2,
) -> RagAnswer:
    """Run RAG for a task-oriented prompt such as flashcards, cheat sheets, or rapid-fire questions."""
    task_text = task_text.strip()
    if not task_text:
        raise ValueError("Please provide a task or prompt.")

    chunks = retrieve_chunks(vectorstore, task_text, domain=domain, top_k=top_k)

    if not chunks:
        return RagAnswer(
            answer="I couldn't find enough material in the selected domain to complete this task.",
            sources=[],
            retrieved_chunks=[],
        )

    context = format_context(chunks)

    try:
        answer_text = call_llm(context, task_text)
    except Exception as exc:
        raise RuntimeError(f"The AI model could not be reached: {exc}") from exc

    return RagAnswer(
        answer=answer_text,
        sources=extract_sources(chunks),
        retrieved_chunks=chunks,
    )
