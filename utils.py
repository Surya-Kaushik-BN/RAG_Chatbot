"""
utils.py
--------
Small, reusable helper functions used across the project.

Keeping these here avoids duplicating logic (like folder/file checks)
in both ingest.py and app.py.
"""

import os
import streamlit as st
from pathlib import Path
from typing import List

from dotenv import load_dotenv

# Project-level constants (kept in one place so paths are consistent everywhere)
BASE_DIR = Path(__file__).resolve().parent
DOCUMENTS_DIR = BASE_DIR / "documents"
CHROMA_DIR = BASE_DIR / "chroma_db"
COLLECTION_NAME = "interview_prep"


def get_domain_for_path(path: Path) -> str:
    """Infer the domain from the PDF path or filename."""
    normalized = path.name.lower()
    parent_names = {part.lower() for part in path.parts}

    if "finance" in parent_names or "finance" in normalized:
        return "finance"
    if "marketing" in parent_names or "marketing" in normalized:
        return "marketing"
    return "all"


def load_env() -> None:
    """Load variables from the .env file into the environment."""
    load_dotenv(BASE_DIR / ".env")


def get_openrouter_api_key() -> str:
    """
    Read the OpenRouter API key.

    It first checks Streamlit's secrets manager (for cloud deployment) and
    then falls back to environment variables (for local .env files).

    Raises a clear error if it is missing, so the app can show a
    friendly message instead of crashing with a cryptic traceback.
    """
    # First, try to get from Streamlit's secrets management
    if "OPENROUTER_API_KEY" in st.secrets:
        return st.secrets["OPENROUTER_API_KEY"]

    # Fallback for local development using .env file
    api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
    if not api_key:
        raise ValueError(
            "OPENROUTER_API_KEY is missing. Add it to your .env file "
            "(see .env.example)."
        )
    return api_key


def get_openrouter_model() -> str:
    """Read the configured LLM model name, falling back to a sensible default."""
    # First, try to get from Streamlit's secrets management
    if "OPENROUTER_MODEL" in st.secrets:
        return st.secrets["OPENROUTER_MODEL"]

    # Fallback for local development using .env file
    default_model = "deepseek/deepseek-chat-v3.1:free"
    return os.getenv("OPENROUTER_MODEL", default_model).strip()


def get_pdf_files(documents_dir: Path = DOCUMENTS_DIR) -> List[Path]:
    """Return a list of all PDF files inside the documents folder and subfolders."""
    if not documents_dir.exists():
        return []
    return sorted(documents_dir.rglob("*.pdf"), key=lambda path: str(path).lower())


def ensure_folders_exist() -> None:
    """Create the documents/ and chroma_db/ folders if they don't exist yet."""
    DOCUMENTS_DIR.mkdir(exist_ok=True)
    CHROMA_DIR.mkdir(exist_ok=True)
