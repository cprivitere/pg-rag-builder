from unittest.mock import MagicMock, patch

from pgrag.rag.pipeline import _find_matching_summary, _summary_score


def test_summary_score_prefers_wiki_gathering_over_recipe():
    question = "What is the highest level mushroom?"
    wiki_gathering = (
        "Mycology items ranked by required skill level (wiki): 1. Mortaferus Mushroom (95)",
        {"name": "Mycology Wiki Gathering Summary"},
        0.84,
    )
    recipe = (
        "Mycology recipes ranked by skill level: 1. Gruesome Spore Bombs (125)",
        {"name": "Mycology Summary"},
        0.94,
    )
    assert _summary_score(question, *wiki_gathering) > _summary_score(question, *recipe)


def test_summary_score_prefers_gathering_for_gathering_terms():
    question = "Which mushroom is best to pick?"
    gathering = (
        "Foraging items ranked by required skill level: 1. Sprigganberry (0)",
        {"name": "Foraging Wiki Gathering Summary"},
        0.9,
    )
    non_gathering = (
        "Foraging recipes ranked by skill level: 1. Berry Pies (20)",
        {"name": "Foraging Summary"},
        0.7,
    )
    assert _summary_score(question, *gathering) > _summary_score(question, *non_gathering)


def test_summary_score_no_gathering_terms_uses_distance():
    question = "What is the strongest weapon?"
    a = (
        "Weapons ranked by damage: 1. Greatsword (300)",
        {"name": "Weapon Summary"},
        0.5,
    )
    b = (
        "Weapons ranked by damage: 1. Greatsword (300)",
        {"name": "Weapon Summary"},
        0.8,
    )
    assert _summary_score(question, *a) > _summary_score(question, *b)


@patch("pgrag.rag.pipeline.embed_text")
@patch("pgrag.rag.pipeline.chromadb.PersistentClient")
def test_find_matching_summary_picks_best_candidate(mock_client, mock_embed):
    mock_embed.return_value = [0.1] * 128
    mock_col = MagicMock()
    mock_client.return_value.get_collection.return_value = mock_col
    mock_col.query.return_value = {
        "ids": [["summary_mycology", "summary_wiki_mycology"]],
        "documents": [
            [
                "Mycology recipes ranked by skill level: 1. Gruesome Spore Bombs (125)",
                "Mycology items ranked by required skill level (wiki): 1. Mortaferus Mushroom (95)",
            ]
        ],
        "metadatas": [
            [
                {"name": "Mycology Summary", "type": "summary"},
                {"name": "Mycology Wiki Gathering Summary", "type": "summary"},
            ]
        ],
        "distances": [[0.9, 0.85]],
    }

    result = _find_matching_summary("What is the highest level mushroom?")
    assert result is not None
    assert "Mortaferus Mushroom (95)" in result


@patch("pgrag.rag.pipeline.embed_text")
@patch("pgrag.rag.pipeline.chromadb.PersistentClient")
def test_find_matching_summary_empty(mock_client, mock_embed):
    mock_embed.return_value = [0.1] * 128
    mock_col = MagicMock()
    mock_client.return_value.get_collection.return_value = mock_col
    mock_col.query.return_value = {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}

    assert _find_matching_summary("nothing here") is None