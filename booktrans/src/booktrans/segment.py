"""Text normalization and sentence splitting for English source text."""

from __future__ import annotations

import re

_ABBREVIATIONS = {
    "mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st", "vs", "etc", "e.g", "i.e",
    "fig", "figs", "eq", "eqs", "no", "nos", "vol", "vols", "ch", "pp", "p", "ed",
    "eds", "al", "cf", "approx", "dept", "inc", "ltd", "co", "jan", "feb", "mar",
    "apr", "jun", "jul", "aug", "sep", "sept", "oct", "nov", "dec", "u.s", "u.k",
}

# Candidate boundary: terminal punctuation, optional closing quote/bracket, whitespace,
# then something that can start a sentence.
_BOUNDARY = re.compile(r"""([.!?…]+["'”’)\]]*)\s+(?=["'“‘(\[]?[A-Z0-9])""")
_SOFT_BREAK = re.compile(r"(?<=[;:,])\s+")
_WS = re.compile(r"\s+")

MAX_SENTENCE_CHARS = 400


def normalize(text: str) -> str:
    """Collapse whitespace and repair line-break hyphenation left by PDF extraction."""
    text = text.replace("­", "")  # soft hyphen
    text = re.sub(r"(\w)-\s*\n\s*(\w)", r"\1\2", text)
    text = text.replace("ﬁ", "fi").replace("ﬂ", "fl")
    return _WS.sub(" ", text).strip()


def _ends_with_abbreviation(chunk: str) -> bool:
    words = chunk.rstrip(".!?…\"'”’)]").split()
    if not words:
        return False
    last = words[-1].lower().rstrip(".")
    # Single capital initials ("J. R. R. Tolkien") are not sentence ends either.
    return last in _ABBREVIATIONS or (len(last) == 1 and last.isalpha())


def split_sentences(text: str) -> list[str]:
    """Split a paragraph into sentences, keeping long sentences under MAX_SENTENCE_CHARS."""
    text = normalize(text)
    if not text:
        return []
    sentences: list[str] = []
    start = 0
    for match in _BOUNDARY.finditer(text):
        end = match.end(1)
        candidate = text[start:end]
        if _ends_with_abbreviation(candidate):
            continue
        sentences.append(candidate.strip())
        start = match.end()
    tail = text[start:].strip()
    if tail:
        sentences.append(tail)

    result: list[str] = []
    for sentence in sentences:
        result.extend(_split_long(sentence))
    return result


def _split_long(sentence: str) -> list[str]:
    """NMT quality collapses on very long inputs; break them at clause punctuation."""
    if len(sentence) <= MAX_SENTENCE_CHARS:
        return [sentence]
    pieces: list[str] = []
    current = ""
    for part in _SOFT_BREAK.split(sentence):
        if current and len(current) + len(part) + 1 > MAX_SENTENCE_CHARS:
            pieces.append(current)
            current = part
        else:
            current = f"{current} {part}".strip()
    if current:
        pieces.append(current)
    # A clause with no punctuation at all can still be too long; cut on word boundaries.
    final: list[str] = []
    for piece in pieces:
        while len(piece) > MAX_SENTENCE_CHARS:
            cut = piece.rfind(" ", 0, MAX_SENTENCE_CHARS)
            cut = cut if cut > 0 else MAX_SENTENCE_CHARS
            final.append(piece[:cut].strip())
            piece = piece[cut:].strip()
        if piece:
            final.append(piece)
    return final
