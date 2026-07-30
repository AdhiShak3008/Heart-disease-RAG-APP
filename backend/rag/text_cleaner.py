"""Text cleaning utilities for PDF/OCR extracted documents."""

from collections import Counter
import re

# -------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------

NAVIGATION_LINES: set[str] = {
    "MAYO",
    "BY",
    "Diseases & Conditions",
    "Doctors",
    "Departments",
    "Request appointment",
    "Register/Log in",
    "On this page",
    "Diagnosis & Doctors &",
    "Symptoms & Diagnosis & Doctors &",
    "causes treatment departments",
}

END_MARKERS: tuple[str, ...] = ("Show References",)

TIMESTAMP_PATTERNS: tuple[str, ...] = (
    r"\d{1,2}/\d{1,2}/\d{2,4},?\s+\d{1,2}:\d{2}\s*(AM|PM)?",
    r"\d{1,2}:\d{2}\s*(AM|PM)",
)

PAGE_NUMBER_PATTERNS: tuple[str, ...] = (
    r"^\s*\d+\s*/\s*\d+\s*$",
    r"^\s*Page\s+\d+(\s+of\s+\d+)?\s*$",
    r"^\s*\d+\s*$",
)

URL_PATTERN = re.compile(
    r"https?://\S+|www\.\S+|w\s*w\s*w\.\S+",
    re.IGNORECASE,
)

WHITESPACE_PATTERN = re.compile(r"[ \t]+")
BLANK_LINES_PATTERN = re.compile(r"\n{3,}")

REPEATED_LINE_MIN_LENGTH: int = 15
REPEATED_LINE_THRESHOLD: int = 3


# -------------------------------------------------------------------
# Cleaner
# -------------------------------------------------------------------


class DocumentCleaner:
    """Clean OCR/PDF extracted text while preserving document structure."""

    def clean(self, text: str) -> str:
        """Run the complete cleaning pipeline."""

        if not text:
            return ""

        text = self.normalize_newlines(text)
        text = self.remove_timestamps(text)
        text = self.remove_urls(text)
        text = self.remove_page_numbers(text)

        text = self.remove_navigation(text)
        text = self.remove_repeated_lines(text)
        text = self.stop_at_end_marker(text)

        text = self.normalize_whitespace(text)
        text = self.remove_short_noise(text)
        text = self.normalize_blank_lines(text)

        return text.strip()

    # -----------------------------------------------------------------

    def normalize_newlines(self, text: str) -> str:
        return text.replace("\r\n", "\n").replace("\r", "\n").replace("\t", " ")

    # -----------------------------------------------------------------

    def remove_timestamps(self, text: str) -> str:

        for pattern in TIMESTAMP_PATTERNS:
            text = re.sub(
                pattern,
                "",
                text,
                flags=re.IGNORECASE,
            )

        return text

    # -----------------------------------------------------------------

    def remove_urls(self, text: str) -> str:

        return URL_PATTERN.sub("", text)

    # -----------------------------------------------------------------

    def remove_page_numbers(self, text: str) -> str:

        for pattern in PAGE_NUMBER_PATTERNS:
            text = re.sub(
                pattern,
                "",
                text,
                flags=re.MULTILINE | re.IGNORECASE,
            )

        return text

    # -----------------------------------------------------------------

    def remove_navigation(self, text: str) -> str:

        cleaned: list[str] = []

        for line in text.splitlines():

            line = line.strip()

            if not line:
                cleaned.append("")
                continue

            if line in NAVIGATION_LINES:
                continue

            lower = line.lower()

            if lower.startswith("request appointment"):
                continue

            if lower.startswith("register/log"):
                continue

            if lower.startswith("advert"):
                continue

            if "mayo clinic press" in lower:
                continue

            if "special offers" in lower:
                continue

            if "newsletter" in lower:
                continue

            if "free mayo clinic" in lower:
                continue

            if "health letter" in lower:
                continue

            if lower == "assessment":
                continue

            cleaned.append(line)

        return "\n".join(cleaned)

    # -----------------------------------------------------------------

    def remove_repeated_lines(self, text: str) -> str:

        lines: list[str] = [line.strip() for line in text.splitlines() if line.strip()]

        counts = Counter(lines)

        cleaned: list[str] = []

        for line in lines:

            if (
                len(line) >= REPEATED_LINE_MIN_LENGTH
                and counts[line] >= REPEATED_LINE_THRESHOLD
            ):
                continue

            cleaned.append(line)

        return "\n".join(cleaned)

    # -----------------------------------------------------------------

    def stop_at_end_marker(self, text: str) -> str:

        lines: list[str] = []

        for line in text.splitlines():

            if any(marker.lower() in line.lower() for marker in END_MARKERS):
                break

            lines.append(line)

        return "\n".join(lines)

    # -----------------------------------------------------------------

    def normalize_whitespace(self, text: str) -> str:

        lines: list[str] = []

        for line in text.splitlines():

            line = WHITESPACE_PATTERN.sub(" ", line)

            lines.append(line.strip())

        return "\n".join(lines)

    # -----------------------------------------------------------------

    def remove_short_noise(self, text: str) -> str:

        cleaned: list[str] = []

        for line in text.splitlines():

            line = line.strip()

            if not line:
                cleaned.append("")
                continue

            # Preserve headings like "Overview"
            if len(line) < 4 and not line.isupper():
                continue

            cleaned.append(line)

        return "\n".join(cleaned)

    # -----------------------------------------------------------------

    def normalize_blank_lines(self, text: str) -> str:

        return BLANK_LINES_PATTERN.sub("\n\n", text)


# -------------------------------------------------------------------
# Convenience wrapper
# -------------------------------------------------------------------

_cleaner = DocumentCleaner()


def clean_text(text: str) -> str:
    """Convenience wrapper for cleaning document text."""

    return _cleaner.clean(text)
