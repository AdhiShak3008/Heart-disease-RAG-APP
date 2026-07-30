"""PDF loader with automatic OCR fallback."""

from pathlib import Path

import fitz
import pytesseract
from PIL import Image

from backend.rag.config import TESSERACT_PATH
from backend.rag.text_cleaner import clean_text

pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH


def load_pdf(path: Path) -> str:
    """Extract text from PDF using OCR when necessary."""

    document = fitz.open(path)

    full_text = []

    for page in document:

        text = page.get_text().strip()

        # Normal PDF
        if len(text) > 50:
            full_text.append(text)
            continue

        # OCR fallback
        pix = page.get_pixmap(dpi=300)

        image = Image.frombytes(
            "RGB",
            [pix.width, pix.height],
            pix.samples,
        )

        ocr_text = pytesseract.image_to_string(
            image,
            lang="eng",
        )

        full_text.append(ocr_text)

    document.close()

    return clean_text("\n".join(full_text))
