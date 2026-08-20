from pgrag.rag.query_classifier import classify_query, find_entity


def test_highest_detected():
    assert classify_query("what is the highest level cheese?") == "comparison"


def test_lowest_detected():
    assert classify_query("what is the lowest level sword?") == "comparison"


def test_best_detected():
    assert classify_query("what is the best armor?") == "comparison"


def test_worst_detected():
    assert classify_query("what is the worst potion?") == "comparison"


def test_most_detected():
    assert classify_query("what is the most powerful spell?") == "comparison"


def test_least_detected():
    assert classify_query("what is the least useful item?") == "comparison"


def test_maximum_detected():
    assert classify_query("which cheese has maximum skill level?") == "comparison"


def test_minimum_detected():
    assert classify_query("minimum level for cheesemaking?") == "comparison"


def test_top_detected():
    assert classify_query("top 5 armor pieces") == "comparison"


def test_strongest_detected():
    assert classify_query("what is the strongest weapon?") == "comparison"


def test_direct_lookup():
    assert classify_query("what level is statehelm sewer cheese?") == "lookup"


def test_what_level_should_is_lookup():
    """'what level should' must classify as lookup, not entity."""
    assert classify_query("What level should I be for Gazluk Keep?") == "lookup"


def test_what_level_do_i_is_lookup():
    assert classify_query("what level do I need for Unarmed?") == "lookup"


def test_general_query():
    assert classify_query("tell me about grinding levels") == "general"


def test_case_insensitive():
    assert classify_query("What Is The HIGHEST Level?") == "comparison"


def test_leveling_intent_wins_over_comparison_phrasing():
    """'most efficient way to level X' is a how-to on a named skill, not an
    item comparison — it must route to the entity dossier."""
    assert classify_query("What is the most efficient way to level Cheesemaking to level 25?") == "entity"


def test_leveling_how_to_routes_to_entity():
    assert classify_query("How do I level Fishing?") == "entity"


def test_leveling_skill_raise_routes_to_entity():
    assert classify_query("How do I raise my Alchemy skill?") == "entity"


def test_leveling_without_named_entity_stays_comparison():
    """Leveling phrasing with no known entity falls through to comparison."""
    assert classify_query("What is the most efficient way to level up?") == "comparison"


def test_minimum_level_lookup_stays_comparison():
    """'minimum level for X' is a value comparison, not leveling intent."""
    assert classify_query("minimum level for cheesemaking?") == "comparison"


def test_concatenated_entity_name_resolves():
    """'GardeningRelated' is a fused compound whose leading token is the
    Gardening skill entity; it must resolve the same hub as 'Gardening'."""
    hub, dtype = find_entity("Which items are GardeningRelated?")
    assert dtype == "skill"
    assert hub == "skillprofile_Gardening"


def test_concatenated_entity_name_falls_back_to_case():
    """CamelCase concatenation (uppercase continuation) matches; a plain
    lowercase word that merely extends the name must not."""
    assert find_entity("lowercasegardening") == (None, None)
    assert find_entity("AmethystVein")[0] == "item_18033"


def test_prefix_word_not_false_positive():
    """'garden' is a prefix of the Gardening skill name but is not the skill
    itself — the boundary fix must not blow it up into the Gardening entity."""
    assert find_entity("garden") == (None, None)


def test_plural_extension_not_matched():
    """A lowercase trailing extension ('GardeningRelated' vs 'Gardens') must
    not match the entity — only true camelCase continuations do."""
    assert find_entity("gardens") == (None, None)
