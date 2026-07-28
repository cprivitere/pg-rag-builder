from documents.chunking import chunk_document, chunk_all_documents, CHUNK_MAX_CHARS


def test_small_doc_not_chunked():
    doc = {"id": "item_1", "type": "item", "text": "short text", "metadata": {"source": "cdn"}}
    result = chunk_document(doc)
    assert len(result) == 1
    assert result[0] is doc


def test_large_doc_split_into_chunks():
    doc = {"id": "item_1", "type": "item", "text": "para one\n\npara two\n\npara three\n\npara four\n\npara five\n\npara six\n\npara seven\n\npara eight\n\npara nine\n\npara ten", "metadata": {"source": "cdn", "name": "Test"}}
    result = chunk_document(doc, max_chars=20)
    assert len(result) > 1


def test_chunk_ids_suffixed():
    doc = {"id": "item_1", "type": "item", "text": "a\n\nb\n\nc\n\nd\n\ne\n\nf\n\ng\n\nh", "metadata": {"source": "cdn"}}
    result = chunk_document(doc, max_chars=10)
    for i, chunk in enumerate(result):
        assert chunk["id"] == f"item_1_chunk_{i}"


def test_chunks_contain_chunk_metadata():
    doc = {"id": "item_1", "type": "item", "text": "a\n\nb\n\nc\n\nd\n\ne\n\nf", "metadata": {"source": "cdn"}}
    result = chunk_document(doc, max_chars=10)
    assert len(result) >= 2
    for chunk in result:
        assert "chunk_index" in chunk["metadata"]
        assert "chunk_count" in chunk["metadata"]
        assert chunk["metadata"]["chunk_count"] == len(result)


def test_chunk_preserves_metadata():
    doc = {"id": "item_1", "type": "item", "text": "a\n\nb\n\nc\n\nd\n\ne\n\nf\n\ng", "metadata": {"source": "cdn", "table": "items", "name": "Test"}}
    result = chunk_document(doc, max_chars=10)
    for chunk in result:
        assert chunk["metadata"]["source"] == "cdn"
        assert chunk["metadata"]["table"] == "items"
        assert chunk["metadata"]["name"] == "Test"
        assert chunk["type"] == "item"


def test_chunk_text_preserved():
    doc = {"id": "item_1", "type": "item", "text": "first paragraph\n\nsecond paragraph\n\nthird paragraph\n\nfourth paragraph", "metadata": {"source": "cdn"}}
    result = chunk_document(doc, max_chars=30)
    combined = "\n\n".join(c["text"] for c in result)
    assert combined == "first paragraph\n\nsecond paragraph\n\nthird paragraph\n\nfourth paragraph"


def test_single_paragraph_exceeds_max():
    text = "word " * 500
    doc = {"id": "item_1", "type": "item", "text": text.strip(), "metadata": {"source": "cdn"}}
    result = chunk_document(doc, max_chars=100)
    assert len(result) > 1


def test_chunk_all_documents():
    docs = [
        {"id": "a", "type": "item", "text": "short", "metadata": {"source": "cdn"}},
        {"id": "b", "type": "item", "text": "x\n\ny\n\nz\n\n1\n\n2\n\n3\n\n4", "metadata": {"source": "cdn"}},
    ]
    result = chunk_all_documents(docs, max_chars=10)
    assert len(result) > len(docs)


def test_chunk_max_chars_constant():
    assert CHUNK_MAX_CHARS == 800
