from app.ingest import _md_sections, chunk_text, load_documents

def test_md_sections_splits_headings_and_defaults_body():
    text_with_headings = """# Intro
Welcome text.

## Returns
Return policy details.

### Exceptions
Some exceptions."""
    sections = _md_sections(text_with_headings)
    titles = [title for title, _ in sections]

    assert "Intro" in titles
    assert "Returns" in titles
    assert "Exceptions" in titles

    no_heading_text = "This is plain content without markdown headings."
    sections_no_heading = _md_sections(no_heading_text)

    assert len(sections_no_heading) == 1
    assert sections_no_heading[0][0] == "Body"
    assert sections_no_heading[0][1] == no_heading_text

def test_chunk_text_respects_size_and_overlap():
    text = "t0 t1 t2 t3 t4 t5 t6 t7 t8 t9"
    chunks = chunk_text(text, chunk_size=4, overlap=2)

    assert chunks == [
        "t0 t1 t2 t3",
        "t2 t3 t4 t5",
        "t4 t5 t6 t7",
        "t6 t7 t8 t9",
    ]

    # Overlap continuity: tail(overlap) of previous == head(overlap) of next
    overlap = 2
    for prev, curr in zip(chunks, chunks[1:]):
        prev_tail = prev.split()[-overlap:]
        curr_head = curr.split()[:overlap]
        assert prev_tail == curr_head

def test_load_documents_filters_only_md_and_txt(tmp_path):
    # Supported files
    (tmp_path / "policy.md").write_text("# Title\nPolicy content", encoding="utf-8")
    (tmp_path / "notes.txt").write_text("Plain text notes", encoding="utf-8")

    # Unsupported files
    (tmp_path / "manual.pdf").write_text("fake pdf", encoding="utf-8")
    (tmp_path / "image.png").write_text("fake png", encoding="utf-8")

    docs = load_documents(str(tmp_path))
    titles = {d["title"] for d in docs}

    assert "policy.md" in titles
    assert "notes.txt" in titles
    assert "manual.pdf" not in titles
    assert "image.png" not in titles

    # Ensure every loaded doc came from allowed extensions only
    assert all(t.lower().endswith((".md", ".txt")) for t in titles)
    assert len(docs) > 0
    assert all("text" in d and isinstance(d["text"], str) for d in docs)
    assert all("section" in d and isinstance(d["section"], str) for d in docs)
    assert all("title" in d and isinstance(d["title"], str) for d in docs)