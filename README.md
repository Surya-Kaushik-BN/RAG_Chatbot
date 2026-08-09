# Finance & Marketing Interview Assistant

A simple Retrieval-Augmented Generation (RAG) chatbot built for an MBA
Product Management course project. It helps students prepare for Finance
and Marketing interviews by answering questions **only** from a set of
uploaded PDF study materials.

## Project Overview

You upload your interview prep PDFs (notes, cheat sheets, frameworks,
formulas) into a `documents/` folder. The app splits them into small
chunks, converts each chunk into an embedding, and stores them in a local
ChromaDB vector database. When you ask a question in the Streamlit chat,
the app retrieves the most relevant chunks and asks an LLM (via
OpenRouter) to answer using only that retrieved content. If the answer
isn't in your documents, the assistant says so instead of guessing.

This keeps the assistant grounded in your material — it will not answer
general trivia or make things up.

## Architecture

```
PDFs
  |
  v
PyPDFLoader
  |
  v
Chunking (RecursiveCharacterTextSplitter, 800 / 150 overlap)
  |
  v
SentenceTransformer Embeddings (all-MiniLM-L6-v2)
  |
  v
ChromaDB (persistent vector store)
  |
  v
Retriever (top 4 similar chunks)
  |
  v
OpenRouter LLM (e.g. deepseek/deepseek-chat-v3.1:free)
  |
  v
Streamlit Chatbot
```

## Installation

### 1. Clone / download the project

Make sure you have the following files and folders:

```
project/
  app.py
  ingest.py
  rag.py
  prompts.py
  utils.py
  requirements.txt
  .env.example
  README.md
  documents/
  chroma_db/
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

Activate it:

- macOS / Linux: `source venv/bin/activate`
- Windows: `venv\Scripts\activate`

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

## Getting an OpenRouter API Key

1. Go to [openrouter.ai](https://openrouter.ai) and create a free account.
2. Open **Keys** in your account settings and create a new API key.
3. Copy the key — you'll paste it into your `.env` file in the next step.

The default model (`deepseek/deepseek-chat-v3.1:free`) is free to use on
OpenRouter, so no payment is required to get started.

## Configuring `.env`

Copy the example file and fill in your key:

```bash
cp .env.example .env
```

Then edit `.env`:

```
OPENROUTER_API_KEY=your-key-here
OPENROUTER_MODEL=deepseek/deepseek-chat-v3.1:free
```

You can swap `OPENROUTER_MODEL` for any other model available on
OpenRouter without changing any code.

## Ingesting PDFs

1. Place your interview prep PDFs inside the `documents/` folder.
2. Run:

```bash
python ingest.py
```

This reads every PDF, splits it into chunks, creates embeddings, and
stores them in `chroma_db/`. It's safe to re-run this after adding new
PDFs — already-ingested chunks are skipped, not duplicated.

> Tip: To enable domain filtering in the app, place finance and marketing
> materials in separate folders like `documents/finance/` and
> `documents/marketing/`, or include `finance` / `marketing` in the PDF
> filename.

## Launching the App

```bash
streamlit run app.py
```

Streamlit will open the chatbot in your browser (usually at
`http://localhost:8501`).

## Example Questions

- Explain Porter's Five Forces
- What is SWOT Analysis?
- Difference between NPV and IRR
- Explain CAPM
- What is STP Marketing?
- What is EBITDA?
- Explain Working Capital
- Tell me about the STAR interview framework

## Project Folder Structure

```
project/
  app.py            # Streamlit UI
  ingest.py         # PDF -> chunks -> embeddings -> ChromaDB
  rag.py            # Retrieval + LLM call (the RAG pipeline)
  prompts.py        # System prompt and prompt templates
  utils.py          # Shared helper functions and constants
  requirements.txt  # Python dependencies
  .env.example      # Template for environment variables
  README.md         # This file
  documents/        # Put your interview prep PDFs here
  chroma_db/        # Auto-created vector database (do not edit by hand)
```

## Troubleshooting

**"No PDFs found in the documents/ folder"**
Add at least one PDF to `documents/` and run `python ingest.py`.

**"Your document database is empty"**
You haven't run `python ingest.py` yet, or it found no readable pages.

**"OPENROUTER_API_KEY is missing"**
Make sure you created a `.env` file (not just `.env.example`) and pasted
in a valid key.

**"The AI model could not be reached"**
Check your internet connection and that your API key is valid. Some
OpenRouter models also have rate limits on the free tier — try again in
a moment or switch `OPENROUTER_MODEL` to another available model.

**Answers say "I couldn't find this information..." too often**
Your question may not be covered in the uploaded PDFs, or the wording is
very different from the source material. Try rephrasing, or add more
detailed PDFs to `documents/` and re-run `python ingest.py`.

**Ingestion seems to skip everything on a re-run**
That's expected — it means those chunks were already added previously.
Only genuinely new or changed content gets embedded again.
