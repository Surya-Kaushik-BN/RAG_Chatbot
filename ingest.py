"""
ingest.py
---------
Run this script whenever you add or update PDFs in the documents/ folder.

What it does, step by step:
1. Loads every PDF in documents/ using PyPDFLoader (keeps page numbers).
2. Splits the text into small overlapping chunks (RecursiveCharacterTextSplitter).
3. Turns each chunk into an embedding using a local SentenceTransformer model.
4. Saves the embeddings into a persistent ChromaDB collection.

Re-running this script is safe: each chunk gets a deterministic ID based on
its content, so chunks that were already ingested are skipped instead of
being duplicated.
"""

import hashlib

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

from utils import DOCUMENTS_DIR, CHROMA_DIR, COLLECTION_NAME, get_pdf_files, ensure_folders_exist

# Chunking settings (as specified by the project requirements)
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150

# Embedding model: small, fast, and runs locally (no API key needed for this part)
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"


def make_chunk_id(chunk: Document) -> str:
    """
    Build a stable, unique ID for a chunk based on its source file, page
    number, and text content. Re-ingesting the same PDF produces the same
    IDs, so ChromaDB naturally avoids storing duplicates.
    """
    source = chunk.metadata.get("source", "unknown")
    page = chunk.metadata.get("page", "unknown")
    content_hash = hashlib.md5(chunk.page_content.encode("utf-8")).hexdigest()
    return f"{source}-p{page}-{content_hash}"


def load_pdfs() -> list[Document]:
    """Load every PDF in documents/ and return a flat list of page-level documents."""
    pdf_files = get_pdf_files()
    all_pages: list[Document] = []

    for pdf_path in pdf_files:
        print(f"Loading: {pdf_path.name}")
        try:
            loader = PyPDFLoader(str(pdf_path))
            pages = loader.load()
            all_pages.extend(pages)
            print(f"  -> loaded {len(pages)} page(s)")
        except Exception as exc:
            # Skip invalid/corrupt PDFs but keep processing the rest
            print(f"  !! Could not read {pdf_path.name}: {exc}")

    return all_pages


def split_into_chunks(pages: list[Document]) -> list[Document]:
    """Split page-level documents into smaller overlapping chunks."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunks = splitter.split_documents(pages)
    print(f"Split {len(pages)} page(s) into {len(chunks)} chunk(s)")
    return chunks


def ingest() -> None:
    """Main entry point: load, chunk, embed, and store all PDFs."""
    ensure_folders_exist()

    pdf_files = get_pdf_files()
    if not pdf_files:
        print(
            f"No PDF files found in {DOCUMENTS_DIR}. "
            "Add some interview prep PDFs there and re-run this script."
        )
        return

    print(f"Found {len(pdf_files)} PDF file(s) in {DOCUMENTS_DIR}\n")

    pages = load_pdfs()
    if not pages:
        print("No readable pages were found. Nothing to ingest.")
        return

    chunks = split_into_chunks(pages)

    print(f"\nLoading embedding model ({EMBEDDING_MODEL_NAME})...")
    embeddings = HuggingFaceEmbeddings(model_name=f"sentence-transformers/{EMBEDDING_MODEL_NAME}")

    print(f"Connecting to ChromaDB at {CHROMA_DIR} ...")
    vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(CHROMA_DIR),
    )

    # Figure out which chunks are genuinely new, so we don't create duplicates
    # when this script is run more than once.
    chunk_ids = [make_chunk_id(chunk) for chunk in chunks]
    existing_ids = set(vectorstore.get(ids=chunk_ids)["ids"])

    new_chunks = []
    new_ids = []
    for chunk, chunk_id in zip(chunks, chunk_ids):
        if chunk_id not in existing_ids:
            new_chunks.append(chunk)
            new_ids.append(chunk_id)

    if not new_chunks:
        print("\nNo new chunks to add - everything is already in ChromaDB.")
        return

    print(f"\nAdding {len(new_chunks)} new chunk(s) to ChromaDB "
          f"({len(chunks) - len(new_chunks)} already existed)...")
    vectorstore.add_documents(documents=new_chunks, ids=new_ids)

    print("\nIngestion complete!")
    print(f"Total chunks in collection now: {vectorstore._collection.count()}")


if __name__ == "__main__":
    ingest()
