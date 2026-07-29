def build_prompt(question, context, query_type="general"):
    base = """You are a Project Gorgon game assistant.

Answer the user's question using only the provided context.

Rules:
- Give a complete helpful answer.
- Include relevant names, skills, levels, ingredients, and quantities when available.
- If the user asks about recipes, include the recipe name and ingredients.
- If multiple answers exist, list them.
- If the context does not contain the answer, say you do not know."""

    if query_type == "comparison":
        base += """

COMPARISON QUESTION DETECTED:
- Examine ALL provided context carefully.
- When asked about highest/lowest/best/worst, compare values across all items.
- Identify the item with the extreme value (maximum or minimum).
- Explain your reasoning: "Comparing X, Y, and Z... the highest is..."
- If a summary document is provided, use it as a quick reference."""

    return f"""{base}

Context:

{context}

Question:
{question}
"""
