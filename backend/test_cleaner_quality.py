from pathlib import Path

from backend.rag.pdf_loader import load_pdf
from backend.rag.text_cleaner import clean_text

pdf = Path("backend/data/docs/Mayo/Heart murmurs.pdf")

text = clean_text(load_pdf(pdf))

checks = [
    "www.",
    "http",
    "Request appointment",
    "Register/Log in",
    "Advertisement",
    "Mayo Clinic Press",
    "FREE Mayo Clinic",
    "Health Letter",
]

print("\n========= CLEANER REPORT =========\n")

for item in checks:
    found = item.lower() in text.lower()

    print(f"{item:25} {'❌ FOUND' if found else '✅ REMOVED'}")

print("\n==================================")
