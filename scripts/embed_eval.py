"""Embedder subset eval (SPEC T86).

Deterministic 1.2k-doc subset + 13 sources-derived queries
(data/eval_subset.json), MRR@10 / hit@3 / hit@5 / recall@10 per embedder
vs jina baseline (:8081).

Run:
  uv run python scripts/embed_eval.py --gen-subset   # (re)build subset
  uv run python scripts/embed_eval.py                # jina baseline only
  uv run python scripts/embed_eval.py --sweep        # + spawn candidates on :8085

Candidate servers spawn like the VRAM probe: llama-server --embedding
--pooling mean (uniform naive pooling for every candidate). Baseline runs its
production pooling (jina-v5 requires --pooling last). Docs truncated to 512
chars for uniformity across short-context models. Metrics computed in each
model's own vector space (dims differ — no cross-model vectors).

Writes: data/eval_subset.json, data/embed_eval.json
"""
import argparse
import hashlib
import json
import random
import re
import subprocess
import sys
import time
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "data"
DOCS_PATH = DATA / "documents.json"
SUBSET_PATH = DATA / "eval_subset.json"
OUT_PATH = DATA / "embed_eval.json"

SEED = 42
TARGET = 1200
N_QUERIES = 13
TRUNC = 512
BATCH = 64
PORT = 8085
HOST = "Scamper"
BASELINE_URL = "http://localhost:8081/embedding"

CHUNK_SUFFIX = re.compile(r"_chunk_\d+$")

SLOTS = [
    ("recipe", "How much XP does {name} give?"),
    ("recipe", "What do I need to craft {name}?"),
    ("item", "What does the item {name} do?"),
    ("item", None),
    ("ability", "What does the ability {name} do?"),
    ("skillprofile", "Tell me everything about the {name} skill"),
    ("quest", "How do I start the quest {name}?"),
    ("wiki", "What does the wiki say about {title}?"),
    ("effect", "What effect does {name} have?"),
    ("tsys", "What does {name} grant?"),
    ("attribute", "What does the stat {name} do?"),
    ("itemuse", "What can {name} be used on?"),
    ("source", "What do I learn from {name}?"),
]

JINA_CURRENT = "jina-v5-text-small (current)"


def load_docs():
    with open(DOCS_PATH, encoding="utf-8") as f:
        return json.load(f)


def base_key(doc_id):
    return CHUNK_SUFFIX.sub("", doc_id)


def doc_name(doc):
    meta = doc.get("metadata", {})
    return meta.get("name") or base_key(doc["id"])


def _type_budgets(counts, target):
    weights = {t: (c ** 0.5) for t, c in counts.items()}
    total_w = sum(weights.values())
    budgets = {}
    for t, w in weights.items():
        b = max(3, int(target * w / total_w))
        budgets[t] = min(counts[t], b)
    remaining = target - sum(budgets.values())
    assert remaining >= 0, (counts, budgets)
    rng = random.Random(SEED)
    order = [t for t, _ in sorted(counts.items(), key=lambda kv: kv[1])]
    while remaining:
        progressed = False
        for t in order:
            if remaining and budgets[t] < counts[t]:
                budgets[t] += 1
                remaining -= 1
                progressed = True
        if not progressed:
            break
    return budgets


def _pick_doc(rng, ids_by_type, used, type_):
    pool = [i for i in ids_by_type[type_] if i not in used]
    if not pool:
        return None
    return rng.choice(pool)


def _family_labels(subset_ids, doc_id):
    key = base_key(doc_id)
    return sorted(i for i in subset_ids if base_key(i) == key)


def _produces_recipes(subset_docs_by_id, item_id, item_name):
    if not item_name:
        return []
    found = []
    for did, doc in subset_docs_by_id.items():
        if doc["type"] != "recipe" or did == item_id:
            continue
        text = doc["text"]
        tail = text.partition("Produces:")[2]
        for line in tail.splitlines():
            if line.startswith("-") and item_name.lower() in line.lower():
                found.append(did)
                break
        if len(found) >= 5:
            break
    return sorted(found)


def gen_subset(docs=None, target=TARGET):
    if docs is None:
        docs = load_docs()
    docs = [d for d in docs if d.get("text", "").strip()]
    target = min(target, len(docs))
    counts = {}
    ids_by_type = {}
    for doc in docs:
        t = doc["type"]
        counts[t] = counts.get(t, 0) + 1
        ids_by_type.setdefault(t, []).append(doc["id"])

    budgets = _type_budgets(counts, target)
    rng = random.Random(SEED)
    order = {}
    for t, ids in ids_by_type.items():
        order[t] = rng.sample(ids, budgets[t]) if len(ids) > budgets[t] else list(ids)

    subset_ids = [i for t in order for i in order[t]]
    assert len(subset_ids) == sum(budgets.values())
    assert len(set(subset_ids)) == len(subset_ids)
    by_id = {d["id"]: d for d in docs}

    docs_out = []
    for did in subset_ids:
        d = by_id[did]
        docs_out.append({"id": did, "type": d["type"], "name": doc_name(d), "text": d["text"]})

    queries = []
    used = set()
    attempts = 0
    while len(queries) < N_QUERIES and attempts < 200:
        attempts += 1
        slot = SLOTS[len(queries)]
        type_, template = slot
        pick = _pick_doc(rng, order, used, type_)
        if pick is None:
            if len(queries) < N_QUERIES:
                break
        used.add(pick)
        doc = by_id[pick]
        name = doc_name(doc)
        if type_ == "wiki":
            title = base_key(doc["id"])
            if title.startswith("wiki_"):
                title = title[len("wiki_"):]
            q = template.format(title=title.replace("_", " ").strip() or name)
        elif template is None:
            item_name = name
            recipes = _produces_recipes({d["id"]: d for d in docs_out}, pick, item_name)
            if recipes:
                q = f"What recipes can produce {item_name}?"
                labels = recipes + [pick]
            else:
                q = f"What does the item {name} do?"
                labels = _family_labels(subset_ids, pick)
        else:
            q = template.format(name=name)
            labels = _family_labels(subset_ids, pick)
        if not labels:
            continue
        queries.append({"q": q, "relevant": sorted(labels)})

    if len(queries) != N_QUERIES:
        raise RuntimeError(f"gen failed: {len(queries)}/{N_QUERIES} queries")
    for q in queries:
        assert all(i in by_id for i in q["relevant"])
    return {"docs": docs_out, "queries": queries}


def save_subset(subset):
    SUBSET_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = SUBSET_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(subset, indent=1, ensure_ascii=False), encoding="utf-8")
    tmp.replace(SUBSET_PATH)


def load_subset():
    return json.loads(SUBSET_PATH.read_text(encoding="utf-8"))


def truncate(texts):
    return [t[:TRUNC] for t in texts]


def _post_embed(texts, url):
    import requests

    r = requests.post(url, json={"content": list(texts)}, timeout=300)
    r.raise_for_status()
    data = r.json()
    vecs = []
    for item in data:
        v = item["embedding"][0]
        if not v:
            raise RuntimeError("empty embedding from server")
        vecs.append(v)
    return vecs


def embed_all(texts, url=BASELINE_URL):
    vecs = []
    truncated = truncate(texts)
    for i in range(0, len(truncated), BATCH):
        vecs.extend(_post_embed(truncated[i:i + BATCH], url))
    assert len(vecs) == len(texts), (len(vecs), len(texts))
    return vecs


def _cosine_scores(query, docs):
    import numpy as np

    q = np.asarray(query, dtype=float)
    q = q / np.linalg.norm(q)
    M = np.asarray(docs, dtype=float)
    norms = np.linalg.norm(M, axis=1)
    M = M / norms[:, None]
    return list(M @ q)


def metrics(query_vecs, doc_vecs, labels, k=10):
    total_mrr = 0.0
    hit3 = hit5 = 0
    recall_sum = 0.0
    for qv, rel in zip(query_vecs, labels):
        rel_set = set(rel)
        scores = _cosine_scores(qv, doc_vecs)
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        for rank, idx in enumerate(ranked, start=1):
            if idx in rel_set:
                total_mrr += 1.0 / rank
                break
        top3 = ranked[:3]
        top5 = ranked[:5]
        if any(i in rel_set for i in top3):
            hit3 += 1
        if any(i in rel_set for i in top5):
            hit5 += 1
        recall_sum += len([i for i in ranked if i in rel_set]) / len(rel_set)
    n = len(query_vecs)
    return {
        "mrr10": round(total_mrr / n, 4),
        "hit3": round(hit3 / n, 4),
        "hit5": round(hit5 / n, 4),
        "recall10": round(recall_sum / n, 4),
    }


def spawn_candidate(cand):
    from scripts.embed_vram_probe import _url_args, get_base_url

    cmd = ["llama-server", *_url_args(cand["url"], cand["local"]),
           "--host", HOST, "--port", str(PORT),
           "--embedding", "--pooling", "mean", "-ngl", "99",
           "--log-file", str(DATA / "embed_eval.log")]
    sys.stderr.write(f"  spawning {cand['name']}...\n")
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    base_url = get_base_url()
    try:
        up = False
        for _ in range(300):
            time.sleep(1)
            try:
                if requests_health(base_url):
                    up = True
                    break
            except Exception:
                continue
        if not up:
            return {"name": cand["name"], "ok": False, "note": "start timeout"}
        return {"base_url": base_url}
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except Exception:
            proc.kill()
        time.sleep(2)


def requests_health(base_url):
    import requests

    r = requests.get(f"{base_url}/health", timeout=2)
    return r.status_code == 200


def run_model(subset, url):
    doc_texts = [d["text"] for d in subset["docs"]]
    q_texts = [q["q"] for q in subset["queries"]]
    sys.stderr.write(f"  embedding {len(doc_texts)} docs at {url}...\n")
    doc_vecs = embed_all(doc_texts, url=url)
    query_vecs = embed_all(q_texts, url=url)
    return metrics(query_vecs, doc_vecs, [q["relevant"] for q in subset["queries"]])


def with_deltas(cand_metrics, base):
    out = dict(cand_metrics)
    for k, v in base.items():
        out[f"delta_{k}"] = round(cand_metrics[k] - v, 4)
    return out


def main():
    parser = argparse.ArgumentParser(description="Embedder subset eval (SPEC T86)")
    parser.add_argument("--gen-subset", action="store_true", help="regenerate eval_subset.json")
    parser.add_argument("--sweep", action="store_true", help="spawn + evaluate candidate embedders")
    args = parser.parse_args()

    if args.gen_subset or not SUBSET_PATH.exists():
        sys.stderr.write("building subset...\n")
        save_subset(gen_subset())
    subset = load_subset()
    sys.stderr.write(f"subset: {len(subset['docs'])} docs, {len(subset['queries'])} queries\n")

    base_url = BASELINE_URL
    if not args.sweep:
        sys.stderr.write(f"baseline evaluate at {base_url}...\n")
        baseline = run_model(subset, base_url)
        print(json.dumps({"subset": {"docs": len(subset["docs"]), "queries": len(subset["queries"])},
                          "baseline": baseline}, indent=2))
        OUT_PATH.write_text(json.dumps(
            {"generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
             "subset": {"docs": len(subset["docs"]), "queries": len(subset["queries"])},
             "baseline": baseline}, indent=2), encoding="utf-8")
        return

    from scripts.embed_vram_probe import CANDIDATES

    baseline = run_model(subset, base_url)
    results = {"generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
               "subset": {"docs": len(subset["docs"]), "queries": len(subset["queries"])},
               "baseline": baseline,
               "candidates": {}}
    for cand in CANDIDATES:
        if cand["name"] == JINA_CURRENT:
            continue
        info = spawn_candidate(cand)
        if not info.get("ok", True):
            results["candidates"][cand["name"]] = {"ok": False, "note": info.get("note")}
            continue
        m = run_model(subset, info["base_url"])
        results["candidates"][cand["name"]] = with_deltas(m, baseline)
        sys.stderr.write(f"  {cand['name']}: {m}\n")
    print(json.dumps(results, indent=2))
    OUT_PATH.write_text(json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()