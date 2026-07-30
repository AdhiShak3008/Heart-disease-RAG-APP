"""Recursive text chunking."""

import re

from backend.rag.config import CHUNK_SIZE, CHUNK_OVERLAP

SEPARATORS = [
    "\n\n",  # paragraphs
    ". ",  # sentences
    "\n",  # lines
    " ",  # words
]


def chunk_text(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> list[str]:

    text = re.sub(r"\s+", " ", text).strip()

    if not text:
        return []

    return _recursive_split(text, chunk_size, overlap, SEPARATORS)


def _recursive_split(text, chunk_size, overlap, separators):

    if len(text) <= chunk_size:
        return [text]

    if not separators:
        return _character_split(text, chunk_size, overlap)

    separator = separators[0]

    parts = text.split(separator)

    chunks = []
    current = ""

    for part in parts:

        candidate = part if not current else current + separator + part

        if len(candidate) <= chunk_size:
            current = candidate

        else:

            if current:
                chunks.extend(
                    _recursive_split(
                        current,
                        chunk_size,
                        overlap,
                        separators[1:],
                    )
                )

            current = part

    if current:
        chunks.extend(
            _recursive_split(
                current,
                chunk_size,
                overlap,
                separators[1:],
            )
        )

    return _apply_overlap(chunks, overlap)


def _character_split(text, chunk_size, overlap):

    chunks = []

    start = 0

    while start < len(text):

        end = start + chunk_size

        chunks.append(text[start:end].strip())

        start += chunk_size - overlap

    return chunks


def _apply_overlap(chunks, overlap):

    if overlap <= 0:
        return chunks

    merged = []

    for i, chunk in enumerate(chunks):

        if i == 0:
            merged.append(chunk)
            continue

        prefix = chunks[i - 1][-overlap:]

        merged.append(prefix + chunk)

    return merged
