from pgrag.rag.retriever import _term_overlap, _rerank, RERANK_MULTIPLIER


def test_term_overlap_full_match():
    score = _term_overlap("how to make potion", "how to make potion recipe")
    assert score == 1.0


def test_term_overlap_partial():
    score = _term_overlap("how to make potion", "potion recipe")
    assert score == 0.25


def test_term_overlap_no_match():
    score = _term_overlap("how to make potion", "sword fighting")
    assert score == 0.0


def test_term_overlap_empty_query():
    score = _term_overlap("", "some document text")
    assert score == 0.0


def test_term_overlap_case_insensitive():
    score = _term_overlap("POTION", "potion recipe")
    assert score == 1.0


def test_rerank_returns_correct_count():
    query = "potion making"
    ids = ["a", "b", "c", "d", "e", "f", "g", "h", "i"]
    docs = ["potion recipe", "sword sharpening", "potion brewing tips", "shield block", "potion effects guide", "arrow fletching", "potion history book", "helmet polishing", "potion ingredients list"]
    metas = [{}] * len(docs)
    dists = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]

    result_ids, result_docs, result_metas, result_dists = _rerank(
        query, ids, docs, metas, dists, 3
    )

    assert len(result_ids) == 3
    assert len(result_docs) == 3
    assert len(result_metas) == 3
    assert len(result_dists) == 3


def test_rerank_promotes_term_match():
    query = "potion"
    ids = ["no_match", "has_match"]
    docs = ["sword fighting guide", "potion brewing guide"]
    metas = [{}, {}]
    dists = [0.1, 0.3]

    result_ids, _, _, _ = _rerank(query, ids, docs, metas, dists, 2)

    assert result_ids[0] == "has_match", "doc with term match should rank first"


def test_rerank_with_scored_inputs():
    query = "potion fighting"
    ids = ["match_both", "match_one", "match_none"]
    docs = ["potion fighting guide", "potion brewing guide", "sword shield guide"]
    metas = [{}, {}, {}]
    dists = [0.3, 0.2, 0.1]

    result_ids, _, _, _ = _rerank(query, ids, docs, metas, dists, 3)

    assert result_ids[0] == "match_both", "doc matching both terms ranks first"


def test_rerank_multiplier_constant():
    assert RERANK_MULTIPLIER == 3
