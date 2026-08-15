from pgrag.rag.prompts import build_prompt


def test_comparison_prompt_includes_instructions():
    prompt = build_prompt("what is the highest level cheese?", "context here", query_type="comparison")
    assert "COMPARISON QUESTION DETECTED" in prompt
    assert "compare values" in prompt.lower() or "Compare" in prompt


def test_general_prompt_no_comparison_section():
    prompt = build_prompt("tell me about cheese", "context here", query_type="general")
    assert "COMPARISON QUESTION DETECTED" not in prompt


def test_lookup_prompt_no_comparison_section():
    prompt = build_prompt("what level is statehelm sewer cheese?", "context here", query_type="lookup")
    assert "COMPARISON QUESTION DETECTED" not in prompt


def test_prompt_contains_context():
    prompt = build_prompt("question", "my context", query_type="general")
    assert "my context" in prompt


def test_prompt_contains_question():
    prompt = build_prompt("my question", "context", query_type="general")
    assert "my question" in prompt
