"""Tests for text canonicalization."""

from brr.core.canonicalize import canonicalize
from brr.core.canonicalize import canonicalize_query


def test_nfc_normalization():
    # e + combining acute = NFC é
    text = "caf\u0065\u0301"
    result = canonicalize(text)
    assert "\u00e9" in result


def test_markdown_link_strip():
    text = "See [the docs](https://example.com) for details"
    result = canonicalize(text)
    assert "the docs" in result
    assert "https://example.com" not in result
    assert "[" not in result


def test_markdown_emphasis_strip():
    text = "This is **bold** and *italic* text"
    result = canonicalize(text)
    assert "bold" in result
    assert "italic" in result
    assert "*" not in result


def test_markdown_heading_strip():
    text = "## Section Title\nContent here"
    result = canonicalize(text)
    assert "Section Title" in result
    assert "#" not in result


def test_code_block_collapse():
    lines = [f"line {i}" for i in range(50)]
    text = "```python\n" + "\n".join(lines) + "\n```"
    result = canonicalize(text)
    assert "line 0" in result
    assert "line 19" in result  # last of head 20
    assert "lines omitted" in result
    assert "line 49" in result  # in tail 10


def test_code_block_short_kept():
    text = "```\nshort code\n```"
    result = canonicalize(text)
    assert "short code" in result
    assert "omitted" not in result


def test_url_line_removal():
    text = "some text\nhttps://example.com/path\nmore text"
    result = canonicalize(text)
    assert "some text" in result
    assert "more text" in result
    assert "https://example.com/path" not in result


def test_truncation():
    text = "a" * 3000
    result = canonicalize(text, max_length=100)
    assert len(result) == 100


def test_query_canonicalize_simple():
    result = canonicalize_query("  hello world  ")
    assert result == "hello world"


def test_query_canonicalize_truncation():
    result = canonicalize_query("a" * 1000, max_length=200)
    assert len(result) == 200
