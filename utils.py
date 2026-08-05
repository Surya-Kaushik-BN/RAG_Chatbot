"""
utils.py
--------
Small, reusable helper functions used across the project.

Keeping these here avoids duplicating logic (like folder/file checks)
in both ingest.py and app.py.
"""

import os
from pathlib import Path
from typing import List

from dotenv import load_dotenv

# Project-level constants (kept in one place so paths are consistent everywhere)
BASE_DIR = Path(__file__).resolve().parent
DOCUMENTS_DIR = BASE_DIR / "documents"
CHROMA_DIR = BASE_DIR / "chroma_db"
COLLECTION_NAME = "interview_prep"


def load_env() -> None:
    """Load variables from the .env file into the environment."""
    load_dotenv(BASE_DIR / ".env")


def get_openrouter_api_key() -> str:
    """
    Read the OpenRouter API key from the environment.

    Raises a clear error if it is missing, so the app can show a
    friendly message instead of crashing with a cryptic traceback.
    """
    api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
    if not api_key:
        raise ValueError(
            "OPENROUTER_API_KEY is missing. Add it to your .env file "
            "(see .env.example)."
        )
    return api_key


def get_openrouter_model() -> str:
    """Read the configured LLM model name, falling back to a sensible default."""
    return os.getenv("OPENROUTER_MODEL", "deepseek/deepseek-chat-v3.1:free").strip()


def get_pdf_files(documents_dir: Path = DOCUMENTS_DIR) -> List[Path]:
    """Return a list of all PDF files inside the documents folder."""
    if not documents_dir.exists():
        return []
    return sorted(documents_dir.glob("*.pdf"))


def ensure_folders_exist() -> None:
    """Create the documents/ and chroma_db/ folders if they don't exist yet."""
    DOCUMENTS_DIR.mkdir(exist_ok=True)
    CHROMA_DIR.mkdir(exist_ok=True)
