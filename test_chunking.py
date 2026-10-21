from app.core.chunking import chunk_text


def test_short_text_is_one_chunk():
    assert chunk_text("Hello world. This is short.", 200, 20) == ["Hello world. This is short."]


def test_empty_text_gives_no_chunks():
    assert chunk_text("   \n\n  ", 100, 10) == []


def test_chunks_respect_size_limit_and_cover_all_text():
    sentences = [f"Sentence number {i} talks about topic {i}." for i in range(60)]
    text = " ".join(sentences)
    chunks = chunk_text(text, chunk_size=200, overlap=40)
    assert len(chunks) > 3
    assert all(0 < len(c) <= 200 for c in chunks)
    joined = " ".join(chunks)
    for s in sentences:
        assert s in joined


def test_consecutive_chunks_overlap():
    text = " ".join(f"Fact {i} is important for the audit." for i in range(40))
    chunks = chunk_text(text, chunk_size=150, overlap=50)
    for left, right in zip(chunks, chunks[1:]):
        last_sentence = left.split(". ")[-1].rstrip(".")
        assert last_sentence in right


def test_overlong_sentence_is_hard_split():
    text = "word " * 300
    chunks = chunk_text(text, chunk_size=100, overlap=10)
    assert len(chunks) >= 15
    assert all(len(c) <= 100 for c in chunks)


def test_single_giant_token_is_split_by_characters():
    chunks = chunk_text("x" * 450, chunk_size=100, overlap=0)
    assert [len(c) for c in chunks] == [100, 100, 100, 100, 50]


def test_invalid_arguments():
    for size, overlap in [(0, 0), (100, 100), (100, -1)]:
        try:
            chunk_text("abc", size, overlap)
        except ValueError:
            continue
        raise AssertionError(f"expected ValueError for {(size, overlap)}")
