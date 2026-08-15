import json
import math
import re
from collections import Counter


class BM25:
    def __init__(self, k1=1.2, b=0.75):
        self.k1 = k1
        self.b = b
        self.doc_count = 0
        self.avgdl = 0.0
        self.doc_freq = Counter()
        self.doc_lengths = []
        self.documents = []
        self.df = Counter()

    def index(self, documents):
        self.documents = list(documents)
        self.doc_count = len(self.documents)
        total_length = 0
        self.doc_lengths = []
        self.df = Counter()

        for doc in self.documents:
            tokens = self._tokenize(doc)
            seen = set()
            for t in tokens:
                self.df[t] += 1
                if t not in seen:
                    seen.add(t)
            self.doc_lengths.append(len(tokens))
            total_length += len(tokens)

        self.avgdl = total_length / self.doc_count if self.doc_count else 0.0

    def _tokenize(self, text):
        return re.findall(r"\w+", text.lower())

    def _idf(self, term):
        n = self.df.get(term, 0)
        return math.log((self.doc_count - n + 0.5) / (n + 0.5) + 1.0)

    def search(self, query, k=10):
        query_terms = self._tokenize(query)
        if not query_terms or not self.doc_count:
            return [], []

        tf = Counter(query_terms)
        scores = []

        for i in range(self.doc_count):
            doc_tokens = self._tokenize(self.documents[i])
            doc_tf = Counter(doc_tokens)
            score = 0.0
            for term in query_terms:
                idf = self._idf(term)
                term_freq = doc_tf.get(term, 0)
                numerator = term_freq * (self.k1 + 1)
                denominator = term_freq + self.k1 * (
                    1 - self.b + self.b * self.doc_lengths[i] / self.avgdl
                )
                score += idf * numerator / denominator if denominator else 0.0
            scores.append((score, i))

        scores.sort(key=lambda x: x[0], reverse=True)
        top = scores[:k]
        return [t[1] for t in top], [t[0] for t in top]


def load_bm25_index(path="data/documents.json"):
    with open(path, "r", encoding="utf-8") as f:
        docs = json.load(f)
    texts = [doc["text"] for doc in docs]
    model = BM25()
    model.index(texts)
    return model, docs
