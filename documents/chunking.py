import re

DEFAULT_MAX_CHARS = 1024
OVERLAP_CHARS = 100

TYPE_MAX_CHARS = {
    "item": 512,
    "recipe": 512,
    "wiki": 1024,
}


def _get_max_chars(doc):
    return TYPE_MAX_CHARS.get(doc.get("type", ""), DEFAULT_MAX_CHARS)


def _split_paragraphs(text):
    return [p.strip() for p in text.split("\n\n") if p.strip()]


def _split_sentences(text):
    parts = re.split(r"(?<=[.!?])\s+", text)
    return [s.strip() for s in parts if s.strip()]


def _find_best_split(text, max_chars):
    if len(text) <= max_chars:
        return text, ""

    cut = max_chars
    best = text[:cut].rstrip()

    for end_char in [". ", "! ", "? ", "\n", ", ", " "]:
        idx = text.rfind(end_char, 0, max_chars)
        if idx > max_chars // 2:
            cut = idx + len(end_char)
            best = text[:cut].rstrip()
            break

    remainder = text[cut:].lstrip()
    return best, remainder


def chunk_document(doc, max_chars=None):
    if max_chars is None:
        max_chars = _get_max_chars(doc)

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
                while para:
                    piece, para = _find_best_split(para, max_chars)
                    chunks.append(piece)
                current = []
                current_len = 0
            else:
                current = [para]
                current_len = para_len

    if current:
        chunks.append("\n\n".join(current))

    if len(chunks) == 1:
        return [doc]

    if OVERLAP_CHARS > 0 and len(chunks) > 1:
        chunks = _apply_overlap(chunks)

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


def _apply_overlap(chunks):
    if len(chunks) <= 1:
        return chunks

    overlapped = [chunks[0]]
    for i in range(1, len(chunks)):
        prev = chunks[i - 1]
        overlap_text = prev[-OVERLAP_CHARS:]
        space_idx = overlap_text.find(" ")
        if space_idx > 0:
            overlap_text = overlap_text[space_idx + 1:]
        overlapped.append(overlap_text + " " + chunks[i])

    return overlapped


def chunk_all_documents(documents, max_chars=None):
    result = []
    for doc in documents:
        result.extend(chunk_document(doc, max_chars))
    return result
