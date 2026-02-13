"""Text canonicalization pipeline.

Steps: NFC normalization -> markdown strip -> code collapse -> truncate.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Final


_MAX_TEXT_LENGTH: Final = 2000
_CODE_KEEP_HEAD: Final = 20
_CODE_KEEP_TAIL: Final = 10
_MAX_IMPORT_STREAK: Final = 2
_DEFAULT_QUERY_MAX: Final = 500

# Markdown link: [text](url) -> text
_MD_LINK_RE: Final = re.compile(r"\[([^\]]*)\]\([^)]*\)")
# Markdown emphasis/bold
_MD_EMPHASIS_RE: Final = re.compile(r"(\*{1,3}|_{1,3})(.*?)\1")
# Markdown headings
_MD_HEADING_RE: Final = re.compile(r"^#{1,6}\s+", re.MULTILINE)
# Fenced code blocks
_CODE_BLOCK_RE: Final = re.compile(r"```[^\n]*\n(.*?)```", re.DOTALL)
# Pure URL lines
_URL_LINE_RE: Final = re.compile(r"^\s*https?://\S+\s*$", re.MULTILINE)
# Import-only lines (Python/JS/Rust common patterns)
_IMPORT_RE: Final = re.compile(
    r"^\s*(import |from \S+ import |use |#include |require\(|const .+ = require\()",
    re.MULTILINE,
)


def _nfc(text: str) -> str:
    return unicodedata.normalize("NFC", text)


def _strip_markdown(text: str) -> str:
    stripped = _MD_LINK_RE.sub(r"\1", text)
    stripped = _MD_EMPHASIS_RE.sub(r"\2", stripped)
    return _MD_HEADING_RE.sub("", stripped)


def _collapse_code_block(match: re.Match[str]) -> str:
    code = match.group(1)
    lines = code.splitlines()
    if len(lines) <= _CODE_KEEP_HEAD + _CODE_KEEP_TAIL:
        return code
    head = lines[:_CODE_KEEP_HEAD]
    tail = lines[-_CODE_KEEP_TAIL:]
    omitted = len(lines) - _CODE_KEEP_HEAD - _CODE_KEEP_TAIL
    parts = ["\n".join(head), f"[... {omitted} lines omitted ...]", "\n".join(tail)]
    return "\n".join(parts)


def _collapse_code_blocks(text: str) -> str:
    return _CODE_BLOCK_RE.sub(_collapse_code_block, text)


def _filter_low_signal(text: str) -> str:
    cleaned = _URL_LINE_RE.sub("", text)
    lines = cleaned.splitlines()
    filtered: list[str] = []
    import_streak = 0
    for line in lines:
        if _IMPORT_RE.match(line):
            import_streak += 1
        else:
            import_streak = 0
        if import_streak <= _MAX_IMPORT_STREAK:
            filtered.append(line)
    return "\n".join(filtered)


def _truncate(text: str, max_length: int) -> str:
    if len(text) <= max_length:
        return text
    return text[:max_length]


def canonicalize(text: str, max_length: int = _MAX_TEXT_LENGTH) -> str:
    """Full canonicalization pipeline for document text.

    Returns:
        Canonicalized text with markdown stripped, code collapsed, and length truncated.
    """
    return _truncate(
        _filter_low_signal(_collapse_code_blocks(_strip_markdown(_nfc(text)))),
        max_length,
    )


def canonicalize_query(text: str, max_length: int = _DEFAULT_QUERY_MAX) -> str:
    """Lightweight canonicalization for query text (no markdown/code processing).

    Returns:
        NFC-normalized, stripped, and truncated query text.
    """
    return _truncate(_nfc(text.strip()), max_length)
