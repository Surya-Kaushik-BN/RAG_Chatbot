"""
prompts.py
----------
All prompt text lives here so it's easy to find and tweak without
touching the RAG logic in rag.py.
"""

# The system prompt sets the assistant's persona and hard rules.
# Keeping the rules explicit (context-only, no hallucination) is what
# makes this a "RAG" assistant instead of a general chatbot.
SYSTEM_PROMPT = """You are an experienced MBA interview coach helping students
prepare for Finance and Marketing interviews.

Rules you must always follow:
- Use ONLY the information provided in the "Context" section below unless the
  user explicitly gives a custom example or numbers in the question.
- If the user provides their own numbers or a case scenario, explain the
  process using those numbers while remaining grounded in the context.
- Never invent facts, figures, or examples that are not supported by the
  provided context or the user's explicit input.
- If the context does not contain the answer, respond exactly with:
  "I couldn't find this information in the interview preparation material."
- Keep answers elaborative unless asked by the user to keep them concise.
- Use bullet points whenever they make the answer clearer.
- Explain concepts simply, as if coaching a student before an interview.
- If the context includes a definition, give the definition first, then
  explain further.
- If the context includes formulas, format them cleanly on their own line.
- When asked for examples or cheat-sheet content, draw them from the context
  and make them easy to review.
- When generating flashcards or quiz questions, output structured sections and
  do not reveal the answer before the student has responded.
"""

# The flashcard prompt is used only for the interactive flashcards feature.
FLASHCARD_GENERATION_PROMPT = """Create exactly one flashcard from the provided context.

Topic: {topic}
Difficulty: {difficulty}
Previous terms: {previous_terms}

Rules:
- Use ONLY the information in the provided context.
- Do NOT show the back content until the card is flipped.
- Do NOT generate more than one flashcard.
- Avoid repeating any previous terms.
- Output exactly in this format:
FRONT:
<term or concept>

BACK:
<definition + key explanation + formula/example if relevant>
"""

CHEATSHEET_TOPIC_DISCOVERY_PROMPT = """From the provided context, list the main topics that are actually covered.

Domain: {domain}

Rules:
- Use ONLY the information in the provided context.
- Do not invent topics.
- Output one topic per line.
"""

CHEATSHEET_SUBTOPIC_DISCOVERY_PROMPT = """From the provided context, list the subtopics for this topic.

Domain: {domain}
Topic: {topic}

Rules:
- Use ONLY the information in the provided context.
- Do not invent subtopics.
- Output one subtopic per line.
- If there are no meaningful subtopics, output a single line: None
"""

CHEATSHEET_GENERATION_PROMPT = """Create a focused cheatsheet from the provided context.

Domain: {domain}
Topic: {topic}
Subtopic: {subtopic}

The cheatsheet must include these sections:
- Key concepts
- Definitions
- Frameworks
- Formulas (where relevant)
- Assumptions
- Applications/examples
- Key comparisons
- Common interview questions
- Common mistakes/traps
- Interview takeaways

Rules:
- Use ONLY the information in the provided context.
- Do not invent references.
- If the context does not include a section, state it briefly with best effort.
- At the end, include Sources as a bullet list with any referenced source pages.
"""

# The user prompt template combines the retrieved context with the question.
# Separating context from question helps the LLM distinguish "facts it can use"
# from "what it needs to answer."
USER_PROMPT_TEMPLATE = """Context from interview preparation material:
{context}

Conversation so far:
{conversation}

Question:
{question}

Answer the question using only the context above and the conversation when
relevant, following all the rules from the system prompt.
"""


def build_user_prompt(context: str, question: str, conversation: str = "") -> str:
    """Fill in the user prompt template with the retrieved context, prior
    conversation, and the current question.
    """
    conv = conversation or "None"
    return USER_PROMPT_TEMPLATE.format(context=context, conversation=conv, question=question)


def build_flashcard_generation_prompt(
    topic: str,
    difficulty: str,
    previous_terms: str,
) -> str:
    return FLASHCARD_GENERATION_PROMPT.format(
        topic=topic,
        difficulty=difficulty,
        previous_terms=previous_terms or "None",
    )


def build_cheatsheet_topic_prompt(domain: str) -> str:
    return CHEATSHEET_TOPIC_DISCOVERY_PROMPT.format(domain=domain)


def build_cheatsheet_subtopic_prompt(domain: str, topic: str) -> str:
    return CHEATSHEET_SUBTOPIC_DISCOVERY_PROMPT.format(domain=domain, topic=topic)


def build_cheatsheet_generation_prompt(
    domain: str,
    topic: str,
    subtopic: str,
) -> str:
    return CHEATSHEET_GENERATION_PROMPT.format(
        domain=domain,
        topic=topic,
        subtopic=subtopic or "None",
    )


QUIZ_GENERATION_PROMPT = """Create exactly one interview-style question from the provided context.

Topic: {topic}
Difficulty: {difficulty}
Previous questions: {previous_questions}

Rules:
- Use ONLY the information in the provided context.
- Do not show the answer before the student responds.
- Do not generate more than one question.
- Do not repeat any previous questions.
- Keep the question clear and interview-style.
- After the question, provide the ideal answer and a short rubric.
- Output exactly in this format:
QUESTION:
<question text>

IDEAL ANSWER:
<ideal answer text>

RUBRIC:
<short rubric>
"""


QUIZ_EVALUATION_PROMPT = """Evaluate a student's answer against the ideal answer using the provided context.

Question:
{question}

Ideal answer:
{ideal_answer}

Student answer:
{student_answer}

Instructions:
- Use ONLY the provided context to judge the answer.
- Choose one evaluation label: Correct, Partially Correct, or Incorrect.
- Provide brief feedback on the student's answer.
- Then show the ideal answer clearly.
- Cite the sources used at the end.
- Do not hallucinate or add new facts.
"""


def build_quiz_generation_prompt(
    topic: str,
    difficulty: str,
    previous_questions: str,
) -> str:
    return QUIZ_GENERATION_PROMPT.format(
        topic=topic,
        difficulty=difficulty,
        previous_questions=previous_questions or "None",
    )


def build_quiz_evaluation_prompt(
    question: str,
    ideal_answer: str,
    student_answer: str,
) -> str:
    return QUIZ_EVALUATION_PROMPT.format(
        question=question,
        ideal_answer=ideal_answer,
        student_answer=student_answer,
    )
