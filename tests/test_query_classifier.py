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


def test_capitalized_compound_does_not_overmatch():
    """A known entity name as the prefix of a capitalized compound must NOT
    resolve that entity. camelCase fusion ('StaffCaptain' -> Staff,
    'ArmorPlate' -> Armor effect, 'BashAttack' -> Bash) is a false-positive
    regression; entity detection is strict whole-word only."""
    assert find_entity("StaffCaptain") == (None, None)
    assert find_entity("ArmorPlate") == (None, None)
    assert find_entity("BashAttack") == (None, None)
    assert find_entity("Hammerhead") == (None, None)
    assert find_entity("AgateStone") == (None, None)
    assert find_entity("BarleySoup") == (None, None)


def test_lowercase_word_extension_not_matched():
    """'garden'/'gardens' merely extend the Gardening name with lowercase
    letters; they are not the skill and must not resolve it."""
    assert find_entity("garden") == (None, None)
    assert find_entity("gardens") == (None, None)
    assert find_entity("lowercasegardening") == (None, None)


def test_gardeningrelated_query_classifies_general():
    """'Which items are GardeningRelated?' is a general items query; the fused
    'GardeningRelated' token is not the Gardening skill, so it must not be
    upgraded to an entity route."""
    assert classify_query("Which items are GardeningRelated?") == "general"


def test_aggregation_recipes_use_stays_general():
    """'What recipes use Animal Feces?' is an aggregation over recipes — the
    entity is a filter, not the answer — so it must NOT route to the single
    item dossier. Paired inverse: 'make Orcish Flour' IS the entity target."""
    assert classify_query("What recipes use Animal Feces?") == "general"
    assert classify_query("What recipes can I make with Spider Silk at Tailoring 4?") == "general"
    assert classify_query("What skill and level do I need to make Orcish Flour?") == "entity"


def test_how_do_i_grow_stays_general():
    """'How do I grow Field Mushrooms?' spans skill + wiki, not a single item
    dossier; a detected plural entity must not upgrade it to an entity route."""
    assert classify_query("How do I grow Field Mushrooms?") == "general"


def test_who_gives_quests_in_area_stays_general():
    """'Who gives quests in the Ranalon Den area?' enumerates quests; the area
    alias must not upgrade it to a single quest dossier."""
    assert classify_query("Who gives quests in the Ranalon Den area?") == "general"


def test_lorebook_series_stays_general():
    """Lore series/books are multi-part narrative synthesis with no single-hub
    dossier; detected lorebook entities must not upgrade to entity (or a false
    comparison when a second 'Lore' skill matches)."""
    assert classify_query("What is the Chalice Saga about?") == "general"
    assert classify_query("Tell me about the lore book The Wasted Wishes.") == "general"


def test_amelthyst_compound_resolves_amethyst_item():
    """'AmethystVein' resolves to the real Amethyst item (spelling split),
    not a hallucinated entity."""
    assert find_entity("AmethystVein")[0] == "item_18033"
