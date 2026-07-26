```markdown
# Project Gorgon RAG Builder - SPEC.md

## Project Goal

Build a local RAG (Retrieval Augmented Generation) assistant for Project Gorgon game data.

The system ingests game data from CDN exports and wiki sources, converts them into searchable documents, creates embeddings locally, stores them in ChromaDB, retrieves relevant context, and uses a local LLM to answer player questions.

The goal is a durable knowledge assistant that can answer questions like:

- "How do I become a rabbit?"
- "What recipes use Fairy Honey?"
- "How do I make butter?"
- "Where do I get this item?"
- "What skills are required?"

The system must support frequent game data updates without requiring a full rebuild.

---

# Current Architecture

```

Project Gorgon Data
|
v
loaders/
|
v
Database object
|
v
documents/builder.py
|
v
data/documents.json
|
v
vectorstore/build_index.py
|
v
ChromaDB
|
v
rag/retriever.py
|
v
rag/pipeline.py
|
v
Local LLM (llama.cpp server)

```

---

# Data Flow

## 1. Data Loading

Source:

```

data/cdn/
data/wiki/

```

Currently loaded CDN tables:

- abilities
- items
- recipes
- quests
- skills
- NPCs
- lore
- areas
- attributes
- and others

Database loading is handled by existing loaders.

---

# Document Generation

Location:

```

documents/builder.py

````

Documents are generated from game data.

Current document format:

```json
{
  "id": "item_96",
  "type": "item",
  "text": "Item: Bunny Juice...",
  "metadata": {
    "source": "cdn",
    "table": "items",
    "name": "Bunny Juice (White Fur)",
    "type": "item"
  }
}
````

---

# Metadata Contract

Every document should eventually contain:

```json
{
  "source": "cdn|wiki",
  "table": "items|recipes|quests|...",
  "type": "item|recipe|quest|...",
  "name": "Human readable name"
}
```

Metadata is important because it will later allow:

* filtering
* ranking
* citations
* better answers
* debugging retrieval

---

# Vector Store

Technology:

```
ChromaDB
```

Active database:

```
data/chroma
```

IMPORTANT:

There was an old duplicate database:

```
vectorstore/chroma
```

It has been removed.

All code uses:

```python
chromadb.PersistentClient(
    path="data/chroma"
)
```

---

# Embeddings

Location:

```
embeddings/
```

Embedding server:

```
llama.cpp embedding endpoint
```

Documents are embedded in batches.

Current batch size:

```
1000 documents
```

---

# Index Building

Location:

```
vectorstore/build_index.py
```

Responsibilities:

* Load documents.json
* Connect to Chroma
* Detect existing documents
* Delete removed documents
* Detect changed documents
* Embed changed documents
* Upsert vectors

Current behavior:

```
Document changed:
    re-embed

Document unchanged:
    skip
```

---

# Current Hash System

Location:

```
vectorstore/hashes.py
```

Current implementation:

```python
hash(json.dumps(document))
```

This currently hashes the entire document.

Consequence:

Metadata-only changes trigger re-embedding.

Future improvement:

Hash only:

```python
{
    "id": document["id"],
    "text": document["text"]
}
```

Metadata changes should update Chroma metadata without requiring new embeddings.

---

# Retrieval

Location:

```
rag/
```

Current pipeline:

```
Question
 |
 v
Embedding
 |
 v
Chroma similarity search
 |
 v
Retrieved documents
 |
 v
Prompt construction
 |
 v
Local LLM answer
```

---

# Current LLM

Running with llama.cpp:

Example:

```
llama-server
```

Model:

```
unsloth/gemma-4-26B-A4B-it-qat-GGUF
```

Server:

```
localhost:8080
```

---

# Current Capabilities

Working examples:

## Rabbit transformation

Question:

```
How do I become a rabbit?
```

Answer:

```
You can become a rabbit by using Bunny Juice (White Fur).
This consumable/exotic potion transforms you into a rabbit,
and the transformation is permanent until dispelled.
```

---

## Recipe lookup

Question:

```
What recipes use Fairy Honey?
```

Answer correctly identifies:

```
Trearclaw
```

with ingredients.

---

# Current Known Issues

## 1. Retrieval ranking

Some irrelevant documents appear:

Example:

Question:

```
How do I become a rabbit?
```

Retrieved:

* Bunny Juice ✅
* Rabbit dye recipe ⚠️
* Rabbit textbooks ⚠️

Future improvements:

* metadata filtering
* reranking
* better chunking
* hybrid search

---

## 2. Metadata-only updates cause re-embedding

Planned improvement:

Separate:

```
embedding hash
```

from:

```
metadata hash
```

so metadata updates are cheap.

---

## 3. Retrieval citations

Currently sources show:

```
item_96 {...metadata...}
```

Future:

Display:

```
Bunny Juice (White Fur)
Source: CDN item database
```

---

# Testing

Tests:

```
tests/
```

Run:

```powershell
uv run python -m tests.test_rag
```

Example:

```
Question:
How do I become a rabbit?
```

---

# Development Rules

Important:

1. Make one change at a time.
2. Preserve incremental indexing.
3. Avoid unnecessary full rebuilds.
4. Do not modify multiple architectural layers at once.
5. Prefer durable fixes over quick patches.

---

# Immediate Next Task

Improve indexing efficiency.

Goal:

Metadata changes should not force embeddings.

Change:

Current:

```
document_hash()
    hashes entire document
```

Future:

```
embedding_hash()
    hashes id + text

metadata_hash()
    hashes metadata
```

Then:

* text change -> embed
* metadata change -> update metadata only
* unchanged -> skip
