from pathlib import Path

from backend.rag.pdf_loader import load_pdf

pdf = Path("backend/data/docs/Mayo/Heart murmurs.pdf")

text = load_pdf(pdf)

print(len(text))
print(text[:1000])
