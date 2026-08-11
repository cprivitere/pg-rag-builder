"""VRAM probe for embedder candidates.

For each candidate: start llama-server on :8084 (host=Scamper), warm one
embed, measure TOTAL GPU dedicated-memory delta vs baseline (production stack
on :8080/:8081/:8082 is constant), stop server.

Run: uv run python scripts/embed_vram_probe.py
Writes: data/embed_vram.json
"""
import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

import requests

DATA = Path(__file__).resolve().parent.parent / "data"
PORT = 8084
HOST = "Scamper"

def get_base_url():
    import socket
    try:
        addrs = socket.getaddrinfo(HOST, PORT, socket.AF_INET6, socket.SOCK_STREAM)
        for res in addrs:
            af, socktype, proto, canonname, sa = res
            ip, port_num, flowinfo, scope_id = sa
            if scope_id:
                return f"http://[{ip}%{scope_id}]:{port_num}"
            else:
                return f"http://[{ip}]:{port_num}"
    except Exception:
        pass
    return f"http://{HOST}:{PORT}"

PROBE = "Project Gorgon gardening skill recipe for pig poop fertilizer"

CANDIDATES = [
    {
        "name": "jina-v5-text-small (current)",
        "url": r"F:\AI\models\hub\models--jinaai--jina-embeddings-v5-text-small-retrieval-GGUF\snapshots\78b0ebcb4c870fdfef409e578b65288b49a4fa90\v5-small-retrieval-Q8_0.gguf",
        "local": True,
    },
    {
        "name": "all-MiniLM-L6-v2",
        "url": "second-state/All-MiniLM-L6-v2-Embedding-GGUF:Q8_0",
        "local": False,
    },
    {
        "name": "mxbai-embed-xsmall-v1",
        "url": "twine-network/mxbai-embed-xsmall-v1-Q8_0-GGUF:Q8_0",
        "local": False,
    },
    {
        "name": "bge-small-en-v1.5",
        "url": "ggml-org/bge-small-en-v1.5-Q8_0-GGUF:Q8_0",
        "local": False,
    },
    {
        "name": "embeddinggemma-300m-qat",
        "url": "ggml-org/embeddinggemma-300m-qat-q8_0-GGUF:Q8_0",
        "local": False,
        "pooling": "last",
        "ubatch": "2048",
    },
    {
        "name": "mxbai-embed-large-v1",
        "url": "ChristianAzinn/mxbai-embed-large-v1-gguf:Q8_0",
        "local": False,
    },
    {
        "name": "KaLM-embedding-mini-instruct-v2.5",
        "url": "Aashraf995/KaLM-embedding-multilingual-mini-instruct-v2.5-Q8_0-GGUF:Q8_0",
        "local": False,
    },
    {
        "name": "nomic-embed-text-v1.5",
        "url": "nomic-ai/nomic-embed-text-v1.5-GGUF:Q8_0",
        "local": False,
    },
    {
        "name": "bge-m3 Q8_0",
        "url": "ggml-org/bge-m3-Q8_0-GGUF",
        "local": False,
        "pooling": "last",
        "ubatch": "2048",
    },
    {
        "name": "bge-m3 Q4_K_M",
        "url": "gpustack/bge-m3-GGUF:Q4_K_M",
        "local": False,
        "pooling": "last",
        "ubatch": "2048",
    },
    {
        "name": "jina-v5-nano-retrieval Q8",
        "url": "jinaai/jina-embeddings-v5-text-nano-retrieval-GGUF:Q8_0",
        "local": False,
        "pooling": "last",
    },
    {
        "name": "jina-v5-nano-text-matching Q8",
        "url": "jinaai/jina-embeddings-v5-text-nano-text-matching-GGUF:Q8_0",
        "local": False,
        "pooling": "last",
    },
    {
        "name": "bge-small-en-v1.5 (unsloth f16)",
        "url": "unsloth/bge-small-en-v1.5-GGUF",
        "local": False,
        "pooling": "cls",
        "query_prefix": "Represent this sentence for searching: ",
    },
    {
        "name": "embeddinggemma-300m (unsloth Q8)",
        "url": "unsloth/embeddinggemma-300M-GGUF:Q8_0",
        "local": False,
        "pooling": "last",
        "ubatch": "2048",
    },
]


def _total_vram_mb():
    out = subprocess.run(
        ["pwsh", "-NoProfile", "-Command",
         "(Get-Counter '\\GPU Adapter Memory(*)\\Dedicated Usage' -ErrorAction SilentlyContinue)"
         ".CounterSamples | Measure-Object -Property CookedValue -Sum"],
        capture_output=True, text=True, timeout=60,
    )
    # Output contains ANSI color codes; strip them
    cleaned = re.sub(r'\x1b\[[0-9;]*m', '', out.stdout)
    m = re.search(r"Sum\s*:\s*([\d.]+)", cleaned)
    return (float(m.group(1)) / 1e6) if m else None  # MB


def _url_args(url, local):
    if local or url.endswith(".gguf"):
        return ["-m", url]
    repo, _, ref = url.partition(":")
    return ["-hf", f"{repo}:{ref}"]


def probe_one(cand):
    proc = None
    try:
        base = _total_vram_mb()
        sys.stderr.write(f"  baseline VRAM: {base:.1f} MB\n")
        
        proc = subprocess.Popen(
            ["llama-server", *_url_args(cand["url"], cand["local"]),
             "--host", HOST, "--port", str(PORT),
             "--embedding", "--pooling", cand.get("pooling", "mean"), "-ngl", "99",
             "-c", "8192", "--ubatch-size", cand.get("ubatch", "512"),
             "--log-file", str(DATA / "embed_vram_probe.log")],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        
        # Wait for server to be ready (model load + bind)
        up = False
        base_url = get_base_url()
        for i in range(180):
            time.sleep(1)
            try:
                r = requests.get(f"{base_url}/health", timeout=2)
                if r.status_code == 200:
                    up = True
                    sys.stderr.write(f"  server up after {i+1}s\n")
                    break
            except Exception:
                continue
        
        if not up:
            return {"name": cand["name"], "vram_delta_mb": None, "ok": False, "note": "start timeout"}

        # Warm embed
        r = requests.post(
            f"{base_url}/embedding",
            json={"content": PROBE},
            timeout=60,
        )
        data = r.json()
        items = data if isinstance(data, list) else [data]
        vec = items[0]["embedding"]
        if vec and isinstance(vec[0], list):
            vec = vec[0]
        dim = len(vec)
        time.sleep(3)
        
        peak = _total_vram_mb()
        delta = round(peak - base, 1) if (base and peak) else None
        sys.stderr.write(f"  peak VRAM: {peak:.1f} MB, delta: {delta} MB\n")
        return {"name": cand["name"], "vram_delta_mb": delta, "dim": dim, "ok": r.status_code == 200}
    except Exception as exc:
        return {"name": cand["name"], "vram_delta_mb": None, "ok": False, "note": str(exc)[:160]}
    finally:
        if proc is not None:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except Exception:
                proc.kill()
            time.sleep(2)


def main():
    parser = argparse.ArgumentParser(description="VRAM probe for embedder candidates")
    parser.add_argument("--only", default=None, help="probe only candidates matching this substring")
    args = parser.parse_args()

    existing = {}
    try:
        for r in json.loads((DATA / "embed_vram.json").read_text(encoding="utf-8")):
            existing[r["name"]] = r
    except (FileNotFoundError, json.JSONDecodeError, TypeError):
        existing = {}

    for cand in CANDIDATES:
        if args.only and args.only.lower() not in cand["name"].lower():
            continue
        sys.stderr.write(f"\nProbing {cand['name']}...\n")
        existing[cand["name"]] = probe_one(cand)
    print(json.dumps(list(existing.values()), indent=2))
    (DATA / "embed_vram.json").write_text(
        json.dumps(list(existing.values()), indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()