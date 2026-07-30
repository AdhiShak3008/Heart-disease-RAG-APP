from pathlib import Path

from backend.rag.pdf_loader import load_pdf
from backend.rag.text_cleaner import clean_text

pdf = Path("backend/data/docs/Mayo/Heart murmurs.pdf")

raw_text = load_pdf(pdf)
text = clean_text(raw_text)

print("=" * 80)
print("FIRST 3000 CHARACTERS")
print("=" * 80)
print(text[:3000])

print("\n")
print("=" * 80)
print(f"Total Characters: {len(text):,}")
print(f"Total Lines: {len(text.splitlines())}")
