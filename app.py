"""
app.py
------
Streamlit front-end for the Finance & Marketing Interview Assistant.

This file only handles the UI: layout, chat history, and displaying
results. All the actual RAG logic lives in rag.py so this file stays
easy to read.
"""
import os
import re
import streamlit as st

from rag import (
    answer_question,
    answer_task,
    discover_cheatsheet_subtopics,
    discover_cheatsheet_topics,
    evaluate_rapid_fire_answer,
    generate_cheatsheet,
    generate_flashcard,
    generate_rapid_fire_question,
    load_vectorstore,
)
from utils import get_openrouter_api_key, get_pdf_files, load_env

SUGGESTED_QUESTIONS = [
    "Explain Porter's Five Forces",
    "What is SWOT Analysis?",
    "Difference between NPV and IRR",
    "Explain CAPM",
    "What is STP Marketing?",
    "What is EBITDA?",
    "Explain Working Capital"
]


def init_session_state() -> None:
    """Set up chat history the first time the app runs."""
    if "messages" not in st.session_state:
        st.session_state.messages = []  # list of {"role", "content", "sources"}


def render_sidebar() -> str:
    """Sidebar with domain selection, suggested questions, and clear chat."""
    with st.sidebar:
        st.header("Choose domain")
        selected_domain = st.selectbox(
            "Subject",
            ["All", "Finance", "Marketing"],
            index=0,
            help="Filter chat, flashcards, and cheat sheets by either finance or marketing material.",
        )

        st.divider()
        st.header("Suggested Questions")
        for question in SUGGESTED_QUESTIONS:
            if st.button(question, use_container_width=True):
                st.session_state.pending_question = question

        st.divider()
        if st.button("Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

    return selected_domain


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


def inject_custom_styles() -> None:
    st.markdown(
        """
        <style>
        :root {
            --bg-main: #e8f2ff;
            --surface: #ffffff;
            --surface-soft: #f0f6ff;
            --surface-alt: #eef4ff;
            --primary-blue: #2563eb;
            --primary-blue-dark: #1d4ed8;
            --primary-blue-light: #dbeafe;
            --primary-contrast: #ffffff;
            --border: rgba(37, 99, 235, .18);
            --shadow: 0 18px 40px rgba(37, 99, 235, .12);
        }

        html, body {
            background: var(--bg-main) !important;
        }

        [data-testid="stAppViewContainer"] {
            background: var(--bg-main) !important;
        }

        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, #1e40af 0%, #2563eb 100%) !important;
            color: var(--primary-contrast) !important;
            border-right: 1px solid rgba(255, 255, 255, .12) !important;
        }

        section[data-testid="stSidebar"] .css-1d391kg,
        section[data-testid="stSidebar"] .css-1d391kg * {
            background: transparent !important;
            color: #f8f9ff !important;
        }

        .css-17lntkn, .css-1gkdijx, .css-1d391kg, .css-1e5imcs {
            background: transparent !important;
        }

        .css-1v0mbdj, .css-1950qzw, .css-1psj8ta, .css-1f1f2m6, .css-1170n75 {
            background: var(--surface) !important;
            border: 1px solid rgba(37, 99, 235, .12) !important;
            border-radius: 22px !important;
            box-shadow: var(--shadow) !important;
            padding: 1.2rem !important;
        }

        .stButton>button, .stButton button {
            background: linear-gradient(135deg, var(--primary-blue), #1d4ed8) !important;
            color: var(--primary-contrast) !important;
            border-radius: 14px !important;
            border: 1px solid rgba(255,255,255,.15) !important;
            box-shadow: 0 12px 24px rgba(37, 99, 235, .18) !important;
            padding: 0.85rem 1.3rem !important;
            font-weight: 700 !important;
        }

        .stButton>button:hover, .stButton button:hover {
            transform: translateY(-1px) !important;
            filter: brightness(1.08) !important;
        }

        .stButton>button:focus-visible, .stButton button:focus-visible {
            outline: 2px solid rgba(93, 183, 248, .55) !important;
            outline-offset: 3px !important;
        }

        .stTextInput>div>input, .stSelectbox>div>div>div>div, .stTextArea>div>textarea {
            border-radius: 16px !important;
            border: 1px solid rgba(37, 99, 235, .22) !important;
            background: #f2f7ff !important;
            padding: 0.88rem !important;
            box-shadow: inset 0 1px 2px rgba(37, 99, 235, .08) !important;
        }

        .stTextInput>div>input:focus, .stSelectbox>div>div>div>div:focus, .stTextArea>div>textarea:focus {
            outline: 2px solid rgba(37, 99, 235, .35) !important;
            outline-offset: 2px !important;
        }

        h1, h2, h3, h4, h5, h6 {
            color: #102949 !important;
            font-weight: 700 !important;
        }

        section[data-testid="stSidebar"] h1,
        section[data-testid="stSidebar"] h2,
        section[data-testid="stSidebar"] h3,
        section[data-testid="stSidebar"] h4,
        section[data-testid="stSidebar"] h5,
        section[data-testid="stSidebar"] h6 {
            color: #f8f9ff !important;
        }

        .stMarkdown h1, .stMarkdown h2, .stMarkdown h3, .stMarkdown h4 {
            color: #102949 !important;
            font-weight: 700 !important;
        }

        .stMarkdown h2 { color: #1d4ed8 !important; }
        .stMarkdown h3 { color: #2563eb !important; }

        .stMarkdown, .stMarkdown p, .stMarkdown span, .stMarkdown div {
            color: #1a202c !important;
        }

        p, span, div:not([data-testid="stSidebar"] div) {
            color: #1a202c !important;
        }

        section[data-testid="stSidebar"] p,
        section[data-testid="stSidebar"] span,
        section[data-testid="stSidebar"] div {
            color: #f8f9ff !important;
        }

        section[data-testid="stChatMessage"] .css-1n76uvr,
        section[data-testid="stChatMessage"] .css-1f1f2m6,
        [data-testid="stChatMessage"] > div {
            background: var(--surface) !important;
            border: 1px solid rgba(15, 40, 81, .08) !important;
            border-radius: 22px !important;
            box-shadow: 0 14px 36px rgba(15, 40, 81, .05) !important;
            padding: 1rem !important;
            margin-bottom: 1rem !important;
        }

        [data-testid="stHorizontalBlock"] {
            background: var(--surface-alt) !important;
            border-radius: 22px !important;
            padding: 1.25rem !important;
            border: 1px solid rgba(15, 40, 81, .08) !important;
        }

        div[role="tab"] {
            border-radius: 999px !important;
            border: 1px solid rgba(15, 40, 81, .12) !important;
            background: #eef5ff !important;
            color: #0f315f !important;
            padding: 0.65rem 1rem !important;
            margin-right: 0.35rem !important;
            font-weight: 700 !important;
        }

        div[role="tab"][aria-selected="true"] {
            background: linear-gradient(135deg, var(--primary-blue), var(--primary-blue-dark)) !important;
            color: #ffffff !important;
            border-color: transparent !important;
            box-shadow: 0 16px 30px rgba(37, 99, 235, .18) !important;
        }

        [data-testid="stToolbar"] {
            background: transparent !important;
        }

        .css-1v0mbdj h2, .css-1v0mbdj h3,
        .css-1950qzw h2, .css-1950qzw h3 {
            margin-top: 0 !important;
        }

        .stAlert {
            border-radius: 18px !important;
            padding: 1rem 1.1rem !important;
        }

        .stAlert.success {
            background: rgba(104, 196, 128, .13) !important;
            border-color: rgba(104, 196, 128, .25) !important;
        }

        .stAlert.warning {
            background: rgba(249, 183, 78, .12) !important;
            border-color: rgba(249, 183, 78, .25) !important;
        }

        .stAlert.error {
            background: rgba(239, 114, 101, .12) !important;
            border-color: rgba(239, 114, 101, .25) !important;
        }

        .stAlert.info {
            background: rgba(93, 183, 248, .12) !important;
            border-color: rgba(93, 183, 248, .25) !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_retrieved_context(chunks: list) -> None:
    """Collapsed-by-default expander showing the raw chunks used, for debugging."""
    with st.expander(f"Retrieved Context ({len(chunks)} chunk(s))", expanded=False):
        for i, chunk in enumerate(chunks, start=1):
            st.markdown(f"**Chunk {i}**")
            st.text(chunk.page_content)


def parse_quiz_question_output(text: str) -> dict:
    """Parse the quiz question, ideal answer, and rubric from the LLM response."""
    pattern = re.compile(
        r"QUESTION:\s*(.*?)\s*IDEAL ANSWER:\s*(.*?)\s*RUBRIC:\s*(.*)",
        re.IGNORECASE | re.DOTALL,
    )
    match = pattern.search(text)
    if match:
        return {
            "question": match.group(1).strip(),
            "ideal_answer": match.group(2).strip(),
            "rubric": match.group(3).strip(),
        }

    # Fallback when the exact format is not followed
    return {
        "question": text.strip(),
        "ideal_answer": "",
        "rubric": "",
    }


def parse_flashcard_output(text: str) -> dict:
    """Parse the flashcard front and back from the LLM response."""
    pattern = re.compile(
        r"FRONT:\s*(.*?)\s*BACK:\s*(.*)",
        re.IGNORECASE | re.DOTALL,
    )
    match = pattern.search(text)
    if match:
        return {
            "front": match.group(1).strip(),
            "back": match.group(2).strip(),
        }

    return {
        "front": text.strip(),
        "back": "",
    }


def init_rapid_fire_state() -> None:
    if "rapid_fire_active" not in st.session_state:
        st.session_state.rapid_fire_active = False
    if "rapid_fire_topic" not in st.session_state:
        st.session_state.rapid_fire_topic = ""
    if "rapid_fire_difficulty" not in st.session_state:
        st.session_state.rapid_fire_difficulty = "Easy"
    if "rapid_fire_total" not in st.session_state:
        st.session_state.rapid_fire_total = 5
    if "rapid_fire_index" not in st.session_state:
        st.session_state.rapid_fire_index = 0
    if "rapid_fire_correct" not in st.session_state:
        st.session_state.rapid_fire_correct = 0
    if "rapid_fire_partial" not in st.session_state:
        st.session_state.rapid_fire_partial = 0
    if "rapid_fire_incorrect" not in st.session_state:
        st.session_state.rapid_fire_incorrect = 0
    if "rapid_fire_asked_questions" not in st.session_state:
        st.session_state.rapid_fire_asked_questions = []
    if "rapid_fire_current_question" not in st.session_state:
        st.session_state.rapid_fire_current_question = ""
    if "rapid_fire_current_ideal_answer" not in st.session_state:
        st.session_state.rapid_fire_current_ideal_answer = ""
    if "rapid_fire_current_rubric" not in st.session_state:
        st.session_state.rapid_fire_current_rubric = ""
    if "rapid_fire_current_sources" not in st.session_state:
        st.session_state.rapid_fire_current_sources = []
    if "rapid_fire_question_done" not in st.session_state:
        st.session_state.rapid_fire_question_done = False
    if "rapid_fire_last_feedback" not in st.session_state:
        st.session_state.rapid_fire_last_feedback = ""
    if "rapid_fire_last_status" not in st.session_state:
        st.session_state.rapid_fire_last_status = ""
    if "rapid_fire_answer_text" not in st.session_state:
        st.session_state.rapid_fire_answer_text = ""


def reset_rapid_fire_state() -> None:
    st.session_state.rapid_fire_active = False
    st.session_state.rapid_fire_topic = ""
    st.session_state.rapid_fire_difficulty = "Easy"
    st.session_state.rapid_fire_total = 5
    st.session_state.rapid_fire_index = 0
    st.session_state.rapid_fire_correct = 0
    st.session_state.rapid_fire_partial = 0
    st.session_state.rapid_fire_incorrect = 0
    st.session_state.rapid_fire_asked_questions = []
    st.session_state.rapid_fire_current_question = ""
    st.session_state.rapid_fire_current_ideal_answer = ""
    st.session_state.rapid_fire_current_rubric = ""
    st.session_state.rapid_fire_current_sources = []
    st.session_state.rapid_fire_question_done = False
    st.session_state.rapid_fire_last_feedback = ""
    st.session_state.rapid_fire_last_status = ""
    st.session_state.rapid_fire_answer_text = ""


def init_flashcard_state() -> None:
    if "flashcard_active" not in st.session_state:
        st.session_state.flashcard_active = False
    if "flashcard_topic" not in st.session_state:
        st.session_state.flashcard_topic = ""
    if "flashcard_difficulty" not in st.session_state:
        st.session_state.flashcard_difficulty = "Easy"
    if "flashcard_total" not in st.session_state:
        st.session_state.flashcard_total = 5
    if "flashcard_index" not in st.session_state:
        st.session_state.flashcard_index = 0
    if "flashcard_history" not in st.session_state:
        st.session_state.flashcard_history = []
    if "flashcard_flipped" not in st.session_state:
        st.session_state.flashcard_flipped = False
    if "flashcard_last_error" not in st.session_state:
        st.session_state.flashcard_last_error = ""


def parse_list_items(text: str) -> list[str]:
    items = []
    for line in text.splitlines():
        raw = line.strip()
        if not raw:
            continue
        if raw.lower().startswith("topics:") or raw.lower().startswith("subtopics:"):
            continue
        cleaned = re.sub(r"^[\d\-\*\.\)\s]+", "", raw).strip()
        if cleaned:
            items.append(cleaned)
    return items


def init_cheatsheet_state() -> None:
    if "cheatsheet_domain" not in st.session_state:
        st.session_state.cheatsheet_domain = ""
    if "cheatsheet_topics" not in st.session_state:
        st.session_state.cheatsheet_topics = []
    if "cheatsheet_selected_topic" not in st.session_state:
        st.session_state.cheatsheet_selected_topic = ""
    if "cheatsheet_subtopics" not in st.session_state:
        st.session_state.cheatsheet_subtopics = {}
    if "cheatsheet_selected_subtopic" not in st.session_state:
        st.session_state.cheatsheet_selected_subtopic = ""
    if "cheatsheet_content" not in st.session_state:
        st.session_state.cheatsheet_content = ""
    if "cheatsheet_sources" not in st.session_state:
        st.session_state.cheatsheet_sources = []
    if "cheatsheet_error" not in st.session_state:
        st.session_state.cheatsheet_error = ""


def reset_cheatsheet_state() -> None:
    st.session_state.cheatsheet_domain = ""
    st.session_state.cheatsheet_topics = []
    st.session_state.cheatsheet_selected_topic = ""
    st.session_state.cheatsheet_subtopics = {}
    st.session_state.cheatsheet_selected_subtopic = ""
    st.session_state.cheatsheet_content = ""
    st.session_state.cheatsheet_sources = []
    st.session_state.cheatsheet_error = ""


def reset_flashcard_state() -> None:
    st.session_state.flashcard_active = False
    st.session_state.flashcard_topic = ""
    st.session_state.flashcard_difficulty = "Easy"
    st.session_state.flashcard_total = 5
    st.session_state.flashcard_index = 0
    st.session_state.flashcard_history = []
    st.session_state.flashcard_flipped = False
    st.session_state.flashcard_last_error = ""


def start_flashcard_card(
    vectorstore,
    domain: str,
    topic: str,
    difficulty: str,
) -> None:
    prompt_results = generate_flashcard(
        vectorstore=vectorstore,
        topic=topic,
        difficulty=difficulty,
        previous_terms=[card["front"] for card in st.session_state.flashcard_history],
        domain=domain,
    )
    parsed = parse_flashcard_output(prompt_results.answer)
    card = {
        "front": parsed["front"],
        "back": parsed["back"],
        "sources": prompt_results.sources or [],
    }

    if st.session_state.flashcard_index < len(st.session_state.flashcard_history) - 1:
        st.session_state.flashcard_history[st.session_state.flashcard_index] = card
    else:
        st.session_state.flashcard_history.append(card)

    st.session_state.flashcard_flipped = False
    st.session_state.flashcard_last_error = ""


def start_rapid_fire_question(
    vectorstore,
    domain: str,
    topic: str,
    difficulty: str,
) -> None:
    prompt_results = generate_rapid_fire_question(
        vectorstore=vectorstore,
        topic=topic,
        difficulty=difficulty,
        previous_questions=st.session_state.rapid_fire_asked_questions,
        domain=domain,
    )
    parsed = parse_quiz_question_output(prompt_results.answer)
    st.session_state.rapid_fire_current_question = parsed["question"]
    st.session_state.rapid_fire_current_ideal_answer = parsed["ideal_answer"]
    st.session_state.rapid_fire_current_rubric = parsed["rubric"]
    st.session_state.rapid_fire_current_sources = prompt_results.sources or []
    st.session_state.rapid_fire_question_done = False
    st.session_state.rapid_fire_last_feedback = ""
    st.session_state.rapid_fire_last_status = ""
    st.session_state.rapid_fire_answer_text = ""
    if st.session_state.rapid_fire_current_question:
        st.session_state.rapid_fire_asked_questions.append(
            st.session_state.rapid_fire_current_question
        )


def evaluate_rapid_fire_response(vectorstore, domain: str) -> None:
    result = evaluate_rapid_fire_answer(
        vectorstore=vectorstore,
        question=st.session_state.rapid_fire_current_question,
        ideal_answer=st.session_state.rapid_fire_current_ideal_answer,
        student_answer=st.session_state.rapid_fire_answer_text,
        domain=domain,
    )
    answer_text = result.answer.strip()
    status_match = re.search(r"\b(Correct|Partially Correct|Incorrect)\b", answer_text, re.IGNORECASE)
    status = status_match.group(1).title() if status_match else "Partially Correct"
    st.session_state.rapid_fire_last_status = status
    st.session_state.rapid_fire_last_feedback = answer_text
    st.session_state.rapid_fire_current_sources = result.sources or []
    st.session_state.rapid_fire_question_done = True
    if status == "Correct":
        st.session_state.rapid_fire_correct += 1
    elif status == "Partially Correct":
        st.session_state.rapid_fire_partial += 1
    else:
        st.session_state.rapid_fire_incorrect += 1


def handle_question(question: str, vectorstore, domain: str = "all") -> None:
    """Run the RAG pipeline for a question and render the result in the chat."""
    st.session_state.messages.append({"role": "user", "content": question, "sources": None})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching interview material and generating an answer..."):
            try:
                # Pass the recent conversation so the model can handle follow-ups
                result = answer_question(vectorstore, question, domain=domain, conversation=st.session_state.messages)
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


def handle_task(prompt: str, vectorstore, domain: str = "all") -> None:
    """Run a non-chat task prompt and render the result."""
    with st.spinner("Generating material from the interview resources..."):
        try:
            result = answer_task(vectorstore, prompt, domain=domain)
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


def main() -> None:
    st.set_page_config(page_title="Finance & Marketing Interview Assistant", page_icon="💼")
    load_env()
    if "OPENROUTER_API_KEY" in st.secrets:
        os.environ["OPENROUTER_API_KEY"] = st.secrets["OPENROUTER_API_KEY"]

    if "OPENROUTER_MODEL" in st.secrets:
        os.environ["OPENROUTER_MODEL"] = st.secrets["OPENROUTER_MODEL"]

    init_session_state()
    init_flashcard_state()
    init_cheatsheet_state()
    inject_custom_styles()

    st.title("Finance & Marketing Interview Assistant")
    st.caption("Prepare smarter using your interview material.")

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

    selected_domain = render_sidebar()
    domain_filter = selected_domain.lower()
    if domain_filter not in {"finance", "marketing"}:
        domain_filter = "all"

    tabs = st.tabs(["Chat", "Flashcards", "Rapid Fire", "Cheat Sheet"])

    with tabs[0]:
        st.subheader(f"Chat ({selected_domain})")
        
        # Create a container for scrollable chat history
        chat_container = st.container(height=200, border=False)
        with chat_container:
            render_chat_history()

        # Input section always stays at bottom
        st.divider()
        question = st.chat_input("Ask a finance or marketing interview question...", key="chat_input")
        if "pending_question" in st.session_state:
            question = st.session_state.pop("pending_question")

        if question:
            handle_question(question, vectorstore, domain_filter)

    with tabs[1]:
        st.subheader(f"Flashcards ({selected_domain})")
        init_flashcard_state()

        topic = st.text_input(
            "Topic or concept",
            value=st.session_state.flashcard_topic,
            placeholder="e.g. Valuation, STP, Working Capital",
            key="flashcard_topic",
        )

        difficulty = st.selectbox(
            "Difficulty",
            ["Easy", "Medium", "Hard"],
            index=["Easy", "Medium", "Hard"].index(st.session_state.flashcard_difficulty),
            key="flashcard_difficulty",
        )

        question_count = st.slider(
            "Number of cards",
            min_value=3,
            max_value=10,
            value=st.session_state.flashcard_total,
            key="flashcard_total",
        )

        if not st.session_state.flashcard_active:
            if st.button("Start Flashcards", use_container_width=True, key="start_flashcards"):
                if not topic.strip():
                    st.warning("Please enter a topic to begin the flashcard session.")
                else:
                    st.session_state.flashcard_active = True
                    st.session_state.flashcard_index = 0
                    st.session_state.flashcard_history = []
                    st.session_state.flashcard_flipped = False
                    start_flashcard_card(vectorstore, domain_filter, topic, difficulty)

        else:
            if not st.session_state.flashcard_history:
                st.warning("Could not generate a flashcard. Try a different topic or difficulty.")
            else:
                card = st.session_state.flashcard_history[st.session_state.flashcard_index]
                st.markdown(
                    f"**Progress:** Card {st.session_state.flashcard_index + 1} of {st.session_state.flashcard_total}"
                )

                if not st.session_state.flashcard_flipped:
                    st.markdown("**Front:**")
                    st.markdown(card["front"])
                else:
                    st.markdown("**Back:**")
                    st.markdown(card["back"])
                    if card["sources"]:
                        render_sources(card["sources"])

                controls = st.columns([1, 1, 1])
                if controls[0].button("Previous", use_container_width=True, key="flashcard_prev"):
                    if st.session_state.flashcard_index > 0:
                        st.session_state.flashcard_index -= 1
                        st.session_state.flashcard_flipped = False
                if controls[1].button("Flip", use_container_width=True, key="flashcard_flip"):
                    st.session_state.flashcard_flipped = not st.session_state.flashcard_flipped
                if controls[2].button("Next", use_container_width=True, key="flashcard_next"):
                    if st.session_state.flashcard_index < len(st.session_state.flashcard_history) - 1:
                        st.session_state.flashcard_index += 1
                        st.session_state.flashcard_flipped = False
                    elif st.session_state.flashcard_index + 1 < st.session_state.flashcard_total:
                        st.session_state.flashcard_index += 1
                        start_flashcard_card(vectorstore, domain_filter, topic, difficulty)
                    else:
                        st.success("Flashcard session complete. Restart to review or begin again.")

                if st.button("Restart Flashcards", use_container_width=True, key="flashcard_restart"):
                    reset_flashcard_state()

    with tabs[2]:
        st.subheader(f"Rapid Fire ({selected_domain})")
        init_rapid_fire_state()

        topic = st.text_input(
            "Topic or concept",
            value=st.session_state.rapid_fire_topic,
            placeholder="e.g. Valuation, STP, Working Capital",
        )

        difficulty = st.selectbox(
            "Difficulty",
            ["Easy", "Medium", "Hard"],
            index=["Easy", "Medium", "Hard"].index(st.session_state.rapid_fire_difficulty),
        )

        question_count = st.slider(
            "Number of questions",
            min_value=3,
            max_value=10,
            value=st.session_state.rapid_fire_total,
        )
        st.session_state.rapid_fire_total = question_count

        if not st.session_state.rapid_fire_active:
            if st.button("Start Rapid Fire", use_container_width=True):
                if not topic.strip():
                    st.warning("Please enter a topic to begin the rapid fire session.")
                else:
                    st.session_state.rapid_fire_active = True
                    st.session_state.rapid_fire_index = 1
                    st.session_state.rapid_fire_correct = 0
                    st.session_state.rapid_fire_partial = 0
                    st.session_state.rapid_fire_incorrect = 0
                    st.session_state.rapid_fire_asked_questions = []
                    start_rapid_fire_question(
                        vectorstore,
                        domain_filter,
                        topic,
                        difficulty,
                    )

        else:
            st.markdown(
                f"**Progress:** Question {st.session_state.rapid_fire_index} of {st.session_state.rapid_fire_total} | "
                f"Correct: {st.session_state.rapid_fire_correct} | "
                f"Partial: {st.session_state.rapid_fire_partial} | "
                f"Incorrect: {st.session_state.rapid_fire_incorrect}"
            )

            if st.session_state.rapid_fire_current_question:
                st.markdown("**Question:**")
                st.markdown(st.session_state.rapid_fire_current_question)

                if not st.session_state.rapid_fire_question_done:
                    st.session_state.rapid_fire_answer_text = st.text_area(
                        "Your answer",
                        value=st.session_state.rapid_fire_answer_text,
                        key="rapid_fire_answer_text",
                        height=180,
                    )
                    if st.button("Submit Answer", use_container_width=True):
                        if not st.session_state.rapid_fire_answer_text.strip():
                            st.warning("Please type your answer before submitting.")
                        else:
                            evaluate_rapid_fire_response(vectorstore, domain_filter)
                else:
                    st.markdown(f"**Result:** {st.session_state.rapid_fire_last_status}")
                    st.markdown(st.session_state.rapid_fire_last_feedback)
                    if st.session_state.rapid_fire_current_sources:
                        render_sources(st.session_state.rapid_fire_current_sources)

                    if st.session_state.rapid_fire_index < st.session_state.rapid_fire_total:
                        if st.button("Next Question", use_container_width=True):
                            st.session_state.rapid_fire_index += 1
                            start_rapid_fire_question(
                                vectorstore,
                                domain_filter,
                                topic,
                                difficulty,
                            )
                    else:
                        st.success("Rapid Fire complete! Review your score and restart if you want another session.")
                        if st.button("Restart Rapid Fire", use_container_width=True):
                            reset_rapid_fire_state()
            else:
                st.warning("Could not generate a question. Try changing the topic or difficulty.")

    with tabs[3]:
        st.subheader(f"Cheat Sheet ({selected_domain})")
        init_cheatsheet_state()

        if st.session_state.cheatsheet_domain != domain_filter:
            reset_cheatsheet_state()
            st.session_state.cheatsheet_domain = domain_filter

        if st.button("Discover Topics from uploaded documents", use_container_width=True, key="discover_cheatsheet_topics"):
            topic_result = discover_cheatsheet_topics(vectorstore, domain_filter)
            st.session_state.cheatsheet_topics = parse_list_items(topic_result.answer)
            st.session_state.cheatsheet_selected_topic = ""
            st.session_state.cheatsheet_selected_subtopic = ""
            st.session_state.cheatsheet_content = ""
            st.session_state.cheatsheet_sources = []
            st.session_state.cheatsheet_error = ""

        if not st.session_state.cheatsheet_topics:
            st.info("Click the button above to build a topic list from the uploaded documents.")
        else:
            st.markdown("**Select a topic**")
            selected_topic = st.selectbox(
                "Topic",
                [""] + st.session_state.cheatsheet_topics,
                index=0,
                key="cheatsheet_topic_select",
            )
            st.session_state.cheatsheet_selected_topic = selected_topic

            if selected_topic:
                if selected_topic not in st.session_state.cheatsheet_subtopics:
                    subtopic_result = discover_cheatsheet_subtopics(
                        vectorstore,
                        selected_topic,
                        domain_filter,
                    )
                    subtopics = parse_list_items(subtopic_result.answer)
                    if not subtopics:
                        subtopics = ["None"]
                    st.session_state.cheatsheet_subtopics[selected_topic] = subtopics

                subtopics = st.session_state.cheatsheet_subtopics[selected_topic]
                st.markdown("**Select a subtopic (if available)**")
                selected_subtopic = st.selectbox(
                    "Subtopic",
                    ["None"] + subtopics if "None" not in subtopics else subtopics,
                    index=0,
                    key="cheatsheet_subtopic_select",
                )
                st.session_state.cheatsheet_selected_subtopic = selected_subtopic if selected_subtopic != "None" else ""

                if st.button("Generate Cheatsheet", use_container_width=True, key="generate_cheatsheet"):
                    if not selected_topic:
                        st.warning("Please select a topic first.")
                    else:
                        cheatsheet_result = generate_cheatsheet(
                            vectorstore,
                            domain_filter,
                            selected_topic,
                            st.session_state.cheatsheet_selected_subtopic,
                        )
                        st.session_state.cheatsheet_content = cheatsheet_result.answer
                        st.session_state.cheatsheet_sources = cheatsheet_result.sources or []
                        st.session_state.cheatsheet_error = ""

            if st.session_state.cheatsheet_content:
                st.markdown("---")
                st.markdown(st.session_state.cheatsheet_content)
                if st.session_state.cheatsheet_sources:
                    render_sources(st.session_state.cheatsheet_sources)

            if st.button("Reset Cheat Sheet selection", use_container_width=True, key="reset_cheatsheet"):
                reset_cheatsheet_state()
                st.session_state.cheatsheet_domain = domain_filter


if __name__ == "__main__":
    main()
