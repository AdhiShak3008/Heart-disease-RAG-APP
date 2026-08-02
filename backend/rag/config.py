"""Configuration for RAG pipeline."""

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"

DOCUMENT_DIR = DATA_DIR / "docs"

QDRANT_PATH = DATA_DIR / "qdrant"

COLLECTION_NAME = "heart_disease"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

CHUNK_SIZE = 180

CHUNK_OVERLAP = 30

TOP_K = 5

TESSERACT_PATH = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
