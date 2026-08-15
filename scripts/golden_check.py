import json
import re
import sys
from pathlib import Path

from pgrag.rag.pipeline import ask

GOLDEN_DIR = Path("data/golden")


def normalize(text):
    text = (text or "").lower()
    text = re.sub(r"[^a-z0-9 ]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def check_golden(golden):
    result = ask(golden["question"])
    answer = normalize(result["answer"])
    misses = []
    for variants in golden["facts"]:
        if not any(normalize(v) in answer for v in variants):
            misses.append(variants[0])
    return result, misses


def main():
    total_miss = 0
    for path in sorted(GOLDEN_DIR.glob("*.json")):
        golden = json.loads(path.read_text(encoding="utf-8"))
        result, misses = check_golden(golden)
        status = "PASS" if not misses else "FAIL"
        print("[%s] %s (%s, %s)" % (
            status,
            golden["id"],
            golden["type"],
            result["query_type"],
        ))
        for m in misses:
            print("    MISSING: %s" % m)
        total_miss += len(misses)

    print()
    if total_miss:
        print("FAIL: %d fact(s) missing" % total_miss)
        sys.exit(1)
    print("OK: all golden facts present")


if __name__ == "__main__":
    main()
