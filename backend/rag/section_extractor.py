"""Extract logical sections from cleaned medical documents."""

import re

SECTION_PATTERNS = [
    "Overview",
    "Symptoms",
    "When to see a doctor",
    "Causes",
    "Risk factors",
    "Complications",
    "Prevention",
    "Diagnosis",
    "Treatment",
    "Tests",
    "Medications",
    "Procedures",
    "Lifestyle",
    "Outlook",
]


def extract_sections(text: str) -> list[dict]:
    """
    Split a cleaned document into logical sections.

    Returns
    -------
    [
        {
            "title": "...",
            "content": "..."
        }
    ]
    """

    lines = [line.strip() for line in text.splitlines()]

    sections = []

    current_title = "Introduction"

    current_content = []

    known = {s.lower() for s in SECTION_PATTERNS}

    for line in lines:

        if not line:
            continue

        normalized = re.sub(r"\s+", " ", line).strip()

        if normalized.lower() in known:

            if current_content:

                sections.append(
                    {
                        "title": current_title,
                        "content": "\n".join(current_content).strip(),
                    }
                )

            current_title = normalized

            current_content = []

        else:

            current_content.append(normalized)

    if current_content:

        sections.append(
            {
                "title": current_title,
                "content": "\n".join(current_content).strip(),
            }
        )

    return sections
