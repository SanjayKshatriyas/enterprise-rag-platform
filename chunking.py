"""Sentence-aware text chunking with overlap.

Chunks are built from whole sentences wherever possible so that retrieved passages stay
readable, and consecutive chunks share a small overlap so facts that straddle a boundary
are not lost.
"""
from __future__ import annotations

from app.core.text import split_sentences


def _hard_split(text: str, size: int) -> list[str]:
    """Split an over-long sentence on word boundaries (and, if needed, by characters)."""
    pieces: list[str] = []
    current = ""
    for word in text.split():
        while len(word) > size:  # a single "word" longer than a chunk (e.g. base64)
            if current:
                pieces.append(current)
                current = ""
            pieces.append(word[:size])
            word = word[size:]
        candidate = f"{current} {word}" if current else word
        if len(candidate) > size:
            pieces.append(current)
            current = word
        else:
            current = candidate
    if current:
        pieces.append(current)
    return pieces


def _units(text: str, chunk_size: int) -> list[str]:
    units: list[str] = []
    for paragraph in text.replace("\r\n", "\n").split("\n\n"):
        paragraph = " ".join(paragraph.split())
        if not paragraph:
            continue
        for sentence in split_sentences(paragraph):
            if len(sentence) <= chunk_size:
                units.append(sentence)
            else:
                units.extend(_hard_split(sentence, chunk_size))
    return units


def chunk_text(text: str, chunk_size: int = 800, overlap: int = 150) -> list[str]:
    """Split ``text`` into chunks of at most ``chunk_size`` characters.

    ``overlap`` is the (approximate, sentence-granular) number of characters repeated at
    the start of the next chunk.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if not 0 <= overlap < chunk_size:
        raise ValueError("overlap must be >= 0 and smaller than chunk_size")

    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    for unit in _units(text, chunk_size):
        addition = len(unit) + (1 if current else 0)
        if current and current_len + addition > chunk_size:
            chunks.append(" ".join(current))

            # Carry the tail of the finished chunk over as overlap.
            carry: list[str] = []
            carry_len = 0
            for previous in reversed(current):
                extra = len(previous) + (1 if carry else 0)
                if carry_len + extra > overlap:
                    break
                carry.insert(0, previous)
                carry_len += extra
            current, current_len = carry, carry_len

            addition = len(unit) + (1 if current else 0)
            if current and current_len + addition > chunk_size:
                current, current_len = [], 0
                addition = len(unit)

        current.append(unit)
        current_len += addition

    if current:
        chunks.append(" ".join(current))
    return chunks
