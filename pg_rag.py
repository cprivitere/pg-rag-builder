"""
Project Gorgon RAG Pipe Function for Open WebUI.

Integrates the custom hybrid BM25 + ChromaDB retrieval pipeline
with Open WebUI's chat interface.

Usage:
  1. Import this file via Admin Panel > Functions > Import
  2. Select "Project Gorgon RAG" from the model dropdown
  3. Ask game-related questions — queries route through the custom pipeline
"""

import asyncio
import os
import sys
from pathlib import Path

PG_ROOT = Path(__file__).parent
if str(PG_ROOT) not in sys.path:
    sys.path.insert(0, str(PG_ROOT))

os.chdir(PG_ROOT)

from pydantic import BaseModel, Field

from rag.query_classifier import classify_query
from rag.retriever import retrieve
from rag.prompts import build_prompt
from rag.llm import generate


class Pipe:
    class Valves(BaseModel):
        TOP_K: int = Field(default=3, description="Number of context chunks to retrieve")
        USE_HYBRID: bool = Field(default=True, description="Enable hybrid BM25 + semantic search")
        USE_RERANK: bool = Field(default=True, description="Enable reranking of results")

    def __init__(self):
        self.valves = self.Valves()

    async def pipe(self, body: dict):
        messages = body.get("messages", [])
        if not messages:
            return "No messages provided."

        query = messages[-1].get("content", "")
        if not query.strip():
            return "Empty query."

        query_type = classify_query(query)

        try:
            results = retrieve(
                query,
                count=self.valves.TOP_K,
                hybrid=self.valves.USE_HYBRID,
                rerank=self.valves.USE_RERANK,
                query_type=query_type,
            )
        except Exception as e:
            return f"Retrieval error: {e}"

        docs = results["documents"][0]

        context = "\n\n---\n\n".join(docs)
        prompt = build_prompt(query, context, query_type=query_type)

        try:
            answer = await asyncio.to_thread(generate, prompt)
        except Exception as e:
            return f"LLM error: {e}"

        sources = []
        for doc_id, dist, meta in zip(
            results["ids"][0],
            results["distances"][0],
            results["metadatas"][0],
        ):
            name = meta.get("name", doc_id)
            table = meta.get("table", "unknown")
            sources.append(f"- {name} ({table})")

        source_block = "\n".join(sources) if sources else "No sources found."

        return f"{answer}\n\n---\n**Sources:**\n{source_block}"
