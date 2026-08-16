"""Post-fusion metadata filter (Step 3 of metadata-bm25-eval).

The dense Chroma path supports `$eq/$ne/$gt/$gte/$lt/$lte` natively, but BM25
fused hits bypass the `where` clause, so retrieve() re-filters afterwards.
That post-fusion pass must:
- evaluate operator dicts instead of blind equality (operator dicts were
  previously compared with `==` and matched nothing)
- match delimited (`" | "`-joined) string fields by token membership
- pass operator dicts through to Chroma untouched
"""
from unittest.mock import MagicMock, patch

from pgrag.rag.retriever import _where_matches, _scalar_eq, retrieve


# --- predicate unit tests ---


def test_where_eq_match():
    assert _where_matches({"skill": "Alchemy"}, {"skill": "Alchemy"})
    assert not _where_matches({"skill": "Alchemy"}, {"skill": "Mycology"})
    assert not _where_matches({"skill": "Alchemy"}, {"skill": None})


def test_where_missing_key_fails():
    assert not _where_matches(
        {"skill": "Alchemy"}, {"skill_level_req": 25}
    )


def test_where_lte_gte():
    meta = {"skill_level_req": 25}
    assert _where_matches(meta, {"skill_level_req": {"$lte": 25}})
    assert _where_matches(meta, {"skill_level_req": {"$lte": 30}})
    assert not _where_matches(meta, {"skill_level_req": {"$lte": 24}})
    assert _where_matches(meta, {"skill_level_req": {"$gte": 25}})
    assert not _where_matches(meta, {"skill_level_req": {"$gte": 26}})


def test_where_gt_lt_ne():
    meta = {"skill_level_req": 25}
    assert _where_matches(meta, {"skill_level_req": {"$gt": 24}})
    assert not _where_matches(meta, {"skill_level_req": {"$gt": 25}})
    assert _where_matches(meta, {"skill_level_req": {"$lt": 26}})
    assert not _where_matches(meta, {"skill_level_req": {"$lt": 25}})
    assert _where_matches(meta, {"skill_level_req": {"$ne": 30}})
    assert not _where_matches(meta, {"skill_level_req": {"$ne": 25}})


def test_where_delimited_membership():
    meta = {"ingredients": "Spider Silk | Toadstool Cap"}
    assert _where_matches(meta, {"ingredients": "Spider Silk"})
    assert _where_matches(meta, {"ingredients": "Toadstool Cap"})
    assert not _where_matches(meta, {"ingredients": "Banana"})
    assert not _where_matches(meta, {"ingredients": "Silk"})


def test_scalar_eq_handles_numeric_strings():
    assert _scalar_eq(25, "25")
    assert not _scalar_eq(25, "26")
    assert not _scalar_eq("not-a-number", "5")


def test_where_compound_and_or():
    meta = {"skill": "Alchemy", "skill_level_req": 25}
    assert _where_matches(meta, {
        "$and": [
            {"skill": "Alchemy"},
            {"skill_level_req": {"$gte": 20}},
        ]
    })
    assert not _where_matches(meta, {
        "$and": [{"skill": "Alchemy"}, {"skill_level_req": {"$gte": 30}}]
    })
    assert _where_matches(meta, {
        "$or": [{"skill": "Baking"}, {"skill_level_req": {"$gte": 20}}]
    })
    assert not _where_matches(meta, {
        "$or": [{"skill": "Baking"}, {"skill_level_req": {"$gte": 30}}]
    })


# --- retrieve() pass-through + fused re-filter ---


@patch("pgrag.rag.retriever.embed_text")
@patch("pgrag.rag.retriever.chromadb.PersistentClient")
def test_retrieve_passes_operator_filter_to_chroma(mock_client, mock_embed):
    mock_embed.return_value = [0.1] * 384
    inst = mock_client.return_value
    col = inst.get_collection.return_value
    col.query.return_value = {
        "ids": [["a"]],
        "documents": [["recipe"]],
        "metadatas": [[{"skill_level_req": 25}]],
        "distances": [[0.1]],
    }

    retrieve(
        "crafting question",
        count=3,
        metadata_filter={"skill_level_req": {"$lte": 25}},
    )
    kwargs = col.query.call_args.kwargs
    assert kwargs["where"] == {"skill_level_req": {"$lte": 25}}


@patch("pgrag.rag.retriever.embed_text")
@patch("pgrag.rag.retriever.chromadb.PersistentClient")
@patch("pgrag.rag.bm25.load_bm25_index")
def test_retrieve_hybrid_filters_fused_docs_by_operator(
    mock_load, mock_client, mock_embed
):
    """BM25-only hits dodge Chroma's where; the post-fusion pass must apply
    the operator clause so `$lte` doesn't silently drop them via `==`."""
    mock_embed.return_value = [0.1] * 384
    inst = mock_client.return_value
    col = inst.get_collection.return_value
    col.query.return_value = {
        "ids": [["dense_1"]],
        "documents": [["dense doc"]],
        "metadatas": [[{"skill_level_req": 40}]],
        "distances": [[0.2]],
    }

    bm25_docs = [
        {"id": "bm25_1", "text": "low-level recipe",
         "metadata": {"skill_level_req": 10}},
        {"id": "bm25_2", "text": "high-level recipe",
         "metadata": {"skill_level_req": 50}},
    ]
    fake_model = MagicMock()
    fake_model.search.return_value = ([1, 0], [0.9, 0.8])  # bm25 hits both
    mock_load.return_value = (fake_model, bm25_docs)

    results = retrieve(
        "recipe question",
        count=3,
        metadata_filter={"skill_level_req": {"$lte": 25}},
        hybrid=True,
    )
    kept = results["metadatas"][0]
    assert {"skill_level_req": 10} in kept  # 10 <= 25 -> kept
    assert {"skill_level_req": 40} not in kept  # dense 40 -> dropped
    assert {"skill_level_req": 50} not in kept  # bm25 50 -> dropped