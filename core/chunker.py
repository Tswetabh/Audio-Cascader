"""Streaming punctuation & pause chunker for S2S pipeline.

Splits transcribed text into natural speech clauses for immediate TTS synthesis,
ensuring low time-to-first-byte (TTFB) without waiting for full paragraphs.
"""

from __future__ import annotations

import re
from typing import List, Sequence

DEFAULT_DELIMITERS = (",", ".", "?", "!", ";", ":")


def chunk_text(
    text: str,
    delimiters: Sequence[str] = DEFAULT_DELIMITERS,
    max_words: int = 12,
) -> List[str]:
    """Split text on punctuation marks or word counts while keeping punctuation.

    Args:
        text: Transcribed input string.
        delimiters: Punctuation characters to split on.
        max_words: Maximum words in a single clause before forcing a pause split.

    Returns:
        List of trimmed text chunks ready for TTS synthesis.
    """
    cleaned = text.strip()
    if not cleaned:
        return []

    escaped_delims = "".join(re.escape(d) for d in delimiters)
    pattern = rf"[^{escaped_delims}]+[{escaped_delims}]*"
    chunks = [m.group(0).strip() for m in re.finditer(pattern, cleaned) if m.group(0).strip()]

    final_chunks: List[str] = []
    for chunk in chunks:
        words = chunk.split()
        if len(words) <= max_words:
            final_chunks.append(chunk)
        else:
            for i in range(0, len(words), max_words):
                sub = " ".join(words[i : i + max_words])
                if i + max_words < len(words) and sub[-1] not in delimiters:
                    sub += ","
                final_chunks.append(sub)

    return [c.strip() for c in final_chunks if c.strip()]
