from pathlib import Path

from backend.rag.pdf_loader import load_pdf
from backend.rag.section_extractor import extract_sections

pdf = Path("backend/data/docs/Mayo/Heart murmurs.pdf")

text = load_pdf(pdf)

sections = extract_sections(text)

for section in sections:

    print("=" * 60)

    print(section["title"])

    print()

    print(section["content"][:500])
