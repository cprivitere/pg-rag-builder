from pgrag.rag.query_classifier import classify_query


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


def test_general_query():
    assert classify_query("tell me about grinding levels") == "general"


def test_case_insensitive():
    assert classify_query("What Is The HIGHEST Level?") == "comparison"
