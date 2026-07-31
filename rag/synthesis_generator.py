"""Synthesis generator — use LLM to create curated docs from scattered chunks."""

from rag.llm import generate, LLMServerError


SYNTHESIS_PROMPT = """You are a game assistant for Project Gorgon. The user asked a question but the search results are scattered across multiple sources. 

Synthesize the following information into a clear, comprehensive answer. Create a well-organized document that combines all the relevant information.

User Question: {query}

Scattered Information Sources:
{sources}

Create a comprehensive, well-organized response that:
1. Directly answers the user's question
2. Combines information from all sources
3. Removes redundancy
4. Presents information in a clear, logical order
5. Uses markdown formatting for readability

Response:"""


def synthesize_answer(query: str, results: list) -> str:
    """Synthesize scattered results into a coherent answer.
    
    Args:
        query: Original user query
        results: List of search result documents
        
    Returns:
        Synthesized answer as string
    """
    # Format sources from results
    sources_text = ""
    for i, result in enumerate(results, 1):
        text = result.get("text", "")
        metadata = result.get("metadata", {})
        source = metadata.get("source", "unknown")
        name = metadata.get("name", "Unknown")
        
        sources_text += f"\nSource {i} ({source} - {name}):\n{text[:500]}...\n"
    
    prompt = SYNTHESIS_PROMPT.format(
        query=query,
        sources=sources_text
    )
    
    try:
        response = generate(prompt)
        return response
    except LLMServerError as e:
        # If LLM fails, return a simple concatenation
        return _fallback_synthesis(query, results)


def _fallback_synthesis(query: str, results: list) -> str:
    """Fallback synthesis when LLM is unavailable.
    
    Args:
        query: Original user query
        results: List of search result documents
        
    Returns:
        Simple concatenated answer
    """
    parts = [f"Based on multiple sources, here's what I found about: {query}\n"]
    
    for i, result in enumerate(results, 1):
        text = result.get("text", "")
        metadata = result.get("metadata", {})
        source = metadata.get("source", "unknown")
        name = metadata.get("name", "Unknown")
        
        # Take first 200 chars of each source
        excerpt = text[:200].replace("\n", " ").strip()
        if len(text) > 200:
            excerpt += "..."
        
        parts.append(f"**{name}** ({source}): {excerpt}")
    
    return "\n\n".join(parts)


def create_curated_doc(query: str, results: list) -> dict:
    """Create a curated document from synthesized results.
    
    Args:
        query: Original user query
        results: List of search result documents
        
    Returns:
        Document dict ready for indexing
    """
    synthesized = synthesize_answer(query, results)
    
    # Create doc ID from query
    doc_id = "synthesized_" + query.lower().replace(" ", "_")[:50]
    
    return {
        "id": doc_id,
        "type": "synthesized",
        "text": synthesized,
        "metadata": {
            "source": "synthesized",
            "table": "synthesized",
            "name": f"Synthesized: {query[:50]}"
        }
    }
