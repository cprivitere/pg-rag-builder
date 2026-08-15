import difflib
import json
import re
from pathlib import Path

MIN_TOKEN_LEN = 5
MATCH_CUTOFF = 0.75

_WORD_VOCAB_CACHE = None

TOKEN_RE = re.compile(r"[a-z]+")


def _word_vocab():
    """Lowercase words (len >= MIN_TOKEN_LEN) found in document names and text."""
    global _WORD_VOCAB_CACHE
    path = Path("data/documents.json")
    try:
        mtime = path.stat().st_mtime
    except FileNotFoundError:
        return set()
    if _WORD_VOCAB_CACHE is not None and _WORD_VOCAB_CACHE[0] == mtime:
        return _WORD_VOCAB_CACHE[1]
    docs = json.loads(path.read_text(encoding="utf-8"))
    words = set()
    for doc in docs:
        meta = doc.get("metadata", {}) or {}
        text = " ".join([meta.get("name") or "", doc.get("text") or ""])
        for word in TOKEN_RE.findall(text.lower()):
            if len(word) >= MIN_TOKEN_LEN:
                words.add(word)
    _WORD_VOCAB_CACHE = (mtime, words)
    return words


def correct_query(query):
    """Fuzzy-correct misspelled tokens against known document-name words."""
    vocab = _word_vocab()
    tokens = TOKEN_RE.findall(query.lower())
    corrected = []
    for token in tokens:
        if token in vocab or len(token) < MIN_TOKEN_LEN:
            corrected.append(token)
            continue
        matches = difflib.get_close_matches(token, vocab, n=1, cutoff=MATCH_CUTOFF)
        corrected.append(matches[0] if matches else token)
    return " ".join(corrected)
