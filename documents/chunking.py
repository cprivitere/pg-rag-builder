CHUNK_MAX_CHARS = 800


def _split_paragraphs(text):
    return [p.strip() for p in text.split("\n\n") if p.strip()]


def chunk_document(doc, max_chars=CHUNK_MAX_CHARS):
    text = doc["text"]
    if len(text) <= max_chars:
        return [doc]

    paragraphs = _split_paragraphs(text)
    if not paragraphs:
        return [doc]

    chunks = []
    current = []
    current_len = 0

    for para in paragraphs:
        para_len = len(para)
        if current_len + para_len + 2 <= max_chars:
            current.append(para)
            current_len += para_len + 2
        else:
            if current:
                chunks.append("\n\n".join(current))
            if para_len > max_chars:
                for i in range(0, para_len, max_chars):
                    chunks.append(para[i:i + max_chars])
                current = []
                current_len = 0
            else:
                current = [para]
                current_len = para_len

    if current:
        chunks.append("\n\n".join(current))

    if len(chunks) == 1:
        return [doc]

    result = []
    for i, chunk_text in enumerate(chunks):
        chunk = dict(doc)
        chunk["id"] = f"{doc['id']}_chunk_{i}"
        chunk["text"] = chunk_text
        chunk["metadata"] = dict(doc["metadata"])
        chunk["metadata"]["chunk_index"] = i
        chunk["metadata"]["chunk_count"] = len(chunks)
        result.append(chunk)

    return result


def chunk_all_documents(documents, max_chars=CHUNK_MAX_CHARS):
    result = []
    for doc in documents:
        result.extend(chunk_document(doc, max_chars))
    return result
