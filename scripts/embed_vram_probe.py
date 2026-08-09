"""VRAM probe for embedder candidates.

For each candidate: start llama-server on :8084 (host=Scamper), warm one
embed, measure TOTAL GPU dedicated-memory delta vs baseline (production stack
on :8080/:8081/:8082 is constant), stop server.

Run: uv run python scripts/embed_vram_probe.py
Writes: data/embed_vram.json
"""
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
             "--embedding", "--pooling", "mean", "-ngl", "99",
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
        dim = len(r.json()["embedding"])
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
    results = []
    for cand in CANDIDATES:
        sys.stderr.write(f"\nProbing {cand['name']}...\n")
        results.append(probe_one(cand))
    print(json.dumps(results, indent=2))
    (DATA / "embed_vram.json").write_text(json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()