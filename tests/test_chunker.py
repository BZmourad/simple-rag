"""Tests for the chunker."""

from src.chunker import split_text


def test_short_text_returns_single_chunk():
    assert split_text("hello world", chunk_size=100, chunk_overlap=10) == ["hello world"]


def test_empty_text_returns_no_chunks():
    assert split_text("", chunk_size=100, chunk_overlap=10) == []
    assert split_text("    \n\n   ", chunk_size=100, chunk_overlap=10) == []


def test_chunks_cover_full_text():
    text = "abcdefghij" * 50  # 500 chars
    chunks = split_text(text, chunk_size=100, chunk_overlap=20)
    # Every character must appear in at least one chunk.
    joined = "".join(chunks)
    for i in range(0, len(text), 20):
        assert text[i : i + 20] in joined


def test_overlap_is_respected():
    text = "0123456789" * 30  # 300 chars
    chunks = split_text(text, chunk_size=100, chunk_overlap=30)
    assert len(chunks) >= 2
    # Consecutive chunks should share at least `chunk_overlap` characters
    # when neither is truncated.
    for prev, nxt in zip(chunks, chunks[1:]):
        assert nxt[:15] in prev or prev[-15:] in nxt


def test_deterministic_output():
    text = "The quick brown fox. " * 100
    a = split_text(text, chunk_size=200, chunk_overlap=50)
    b = split_text(text, chunk_size=200, chunk_overlap=50)
    assert a == b


def test_invalid_arguments():
    import pytest

    with pytest.raises(ValueError):
        split_text("hi", chunk_size=0, chunk_overlap=0)
    with pytest.raises(ValueError):
        split_text("hi", chunk_size=100, chunk_overlap=100)
    with pytest.raises(ValueError):
        split_text("hi", chunk_size=100, chunk_overlap=-1)