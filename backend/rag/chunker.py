"""Recursive text chunking."""

from __future__ import annotations

import re

from backend.rag.config import CHUNK_SIZE, CHUNK_OVERLAP

SEPARATORS = (
    "\n\n",   # paragraphs
    ". ",     # sentences
    "\n",     # lines
    " ",      # words
)

MIN_CHUNK_WORDS = 20


def chunk_text(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> list[str]:
    """
    Split text recursively while preserving semantic boundaries.
    """

    text = re.sub(r"\s+", " ", text).strip()

    if not text:
        return []

    chunks = _recursive_split(
        text,
        chunk_size,
        overlap,
        list(SEPARATORS),
    )

    chunks = _merge_small_chunks(chunks)

    return chunks


def _recursive_split(
    text: str,
    chunk_size: int,
    overlap: int,
    separators: list[str],
) -> list[str]:

    if len(text.split()) <= chunk_size:
        return [text.strip()]

    if not separators:
        return _word_split(
            text,
            chunk_size,
            overlap,
        )

    separator = separators[0]

    parts = text.split(separator)

    chunks = []

    current = ""

    for part in parts:

        candidate = part if not current else current + separator + part

        if len(candidate.split()) <= chunk_size:
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

    return _apply_overlap(
        chunks,
        overlap,
    )


def _word_split(
    text: str,
    chunk_size: int,
    overlap: int,
) -> list[str]:

    words = text.split()

    chunks = []

    step = chunk_size - overlap

    for start in range(0, len(words), step):

        end = start + chunk_size

        chunk = " ".join(words[start:end])

        if chunk:
            chunks.append(chunk)

    return chunks


def _apply_overlap(
    chunks: list[str],
    overlap: int,
) -> list[str]:

    if overlap <= 0:
        return chunks

    merged = []

    for i, chunk in enumerate(chunks):

        if i == 0:
            merged.append(chunk)
            continue

        previous_words = chunks[i - 1].split()

        prefix = " ".join(previous_words[-overlap:])

        merged.append(f"{prefix} {chunk}")

    return merged


def _merge_small_chunks(
    chunks: list[str],
) -> list[str]:

    if not chunks:
        return chunks

    merged = []

    for chunk in chunks:

        if (
            merged
            and len(chunk.split()) < MIN_CHUNK_WORDS
        ):

            merged[-1] += " " + chunk

        else:

            merged.append(chunk)

    return merged