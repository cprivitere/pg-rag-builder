import json
from pathlib import Path

import pytest

from scripts.golden_check import check_golden, GOLDEN_DIR

_GOLDEN_FILES = sorted(GOLDEN_DIR.glob("*.json"))


def _servers_up():
    import requests

    for port in (8080, 8081):
        try:
            requests.get(f"http://127.0.0.1:{port}/health", timeout=2)
        except Exception:
            return False
    return True


pytestmark = pytest.mark.skipif(
    not _servers_up(),
    reason="V38: golden check needs LLM (:8080) + embedding (:8081) servers",
)


@pytest.mark.parametrize("path", _GOLDEN_FILES, ids=lambda p: p.stem)
def test_golden_facts(path):
    golden = json.loads(path.read_text(encoding="utf-8"))
    _, misses = check_golden(golden)
    assert misses == [], f"missing facts: {misses}"
