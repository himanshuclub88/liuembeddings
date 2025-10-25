# liuembeddings/utils.py

import re
from typing import List


def clean_text(text: str, lowercase: bool = True, remove_extra_spaces: bool = True) -> str:
    """
    Basic text cleaning: optional lowercasing and removing extra spaces.
    """
    if lowercase:
        text = text.lower()
    if remove_extra_spaces:
        text = re.sub(r"\s+", " ", text).strip()
    return text


def split_text(
    text: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    split_by_sentences: bool = True,
    lowercase: bool = False,
    remove_extra_spaces: bool = False
) -> List[str]:
    """
    Split text into overlapping chunks.

    - If split_by_sentences=True, split after ., !, ? regardless of spaces (e.g., 'Hello.World'). 
    - If False, chunk purely by characters with overlap.
    - Long sentences exceeding chunk_size are sliced with overlap rather than overflowing a chunk.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if not (0 <= chunk_overlap < chunk_size):
        raise ValueError("chunk_overlap must be >= 0 and < chunk_size")

    if lowercase or remove_extra_spaces:
        text = clean_text(text, lowercase, remove_extra_spaces)

    if not text:
        return []

    # Helper to slice a long string into fixed-size windows with overlap.
    def _slice_long(s: str) -> List[str]:
        step = max(1, chunk_size - chunk_overlap)
        out = []
        for i in range(0, len(s), step):
            piece = s[i : i + chunk_size].strip()
            if piece:
                out.append(piece)
        return out

    if split_by_sentences:
        # Split at ., !, ? whether or not there is a following space.
        # Using lookbehind + \s* ensures we split even when there is no space after punctuation.
        sentences = re.split(r"(?<=[.!?])\s*", text)
        sentences = [s for s in sentences if s]  # drop empties
    else:
        sentences = [text]

    chunks: List[str] = []
    current = ""

    for sentence in sentences:
        # If an individual sentence is longer than chunk_size, slice it directly.
        if len(sentence) > chunk_size:
            if current:
                chunks.append(current.strip())
                current = ""
            chunks.extend(_slice_long(sentence))
            continue

        if not current:
            current = sentence
        elif len(current) + 1 + len(sentence) <= chunk_size:
            current = f"{current} {sentence}"
        else:
            # flush current
            chunks.append(current.strip())

            if chunk_overlap > 0:
                # seed next with overlap from the end of the previous chunk
                seed = current[-chunk_overlap:].strip()
                candidate = f"{seed} {sentence}".strip()
                # if overlap + sentence still overflow, start fresh with sentence
                current = sentence if len(candidate) > chunk_size else candidate
            else:
                current = sentence

    if current:
        chunks.append(current.strip())

    return chunks

