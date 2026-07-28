from processors.resolver import GameResolver
from documents.wiki_builder import build_wiki_documents
from documents.chunking import chunk_all_documents


def build_item_documents(db):
    documents = []

    items = db.tables.get("items", {})

    for item_id, item in items.items():

        text = f"""
Item: {item.get('Name', item_id)}

Internal Name:
{item.get('InternalName', '')}

Keywords:
{', '.join(item.get('Keywords', []))}

Usage:
{item.get('Description', '')}

Description:
{item.get('Description', '')}

Stack Size:
{item.get('MaxStackSize', '')}

Value:
{item.get('Value', '')}
"""

        documents.append({
            "id": item_id,
            "type": "item",
            "text": text.strip(),
            "metadata": {
                "source": "cdn",
                "table": "items",
                "name": item.get("Name", item_id)
            }
        })

    return documents


def build_recipe_documents(db):
    documents = []

    resolver = GameResolver(db)

    recipes = db.tables.get("recipes", {})

    for recipe_id, recipe in recipes.items():

        ingredients = []

        for ingredient in recipe.get("Ingredients", []):

            if "ItemCode" in ingredient:
                name = resolver.item_name(
                    ingredient["ItemCode"]
                )

                ingredients.append(
                    f"- {name} x{ingredient.get('StackSize', 1)}"
                )

            elif "ItemKeys" in ingredient:
                desc = ingredient.get("Desc")
                keys = ", ".join(ingredient["ItemKeys"])
                if desc:
                    ingredients.append(
                        f"- {desc} "
                        f"(category: {keys}) "
                        f"x{ingredient.get('StackSize', 1)}"
                    )
                else:
                    ingredients.append(
                        f"- Any item matching: {keys} "
                        f"x{ingredient.get('StackSize', 1)}"
                    )

            else:
                ingredients.append(
                    f"- {ingredient.get('Desc', 'Unknown ingredient')}"
                )

        results = []

        for result in recipe.get("ResultItems", []):

            name = resolver.item_name(
                result["ItemCode"]
            )

            results.append(
                f"- {name} x{result['StackSize']}"
            )

        text = f"""
Recipe: {recipe.get('Name', recipe_id)}

Skill:
{recipe.get('Skill', '')}

Required Skill Level:
{recipe.get('SkillLevelReq', 0)}

Description:
{recipe.get('Description', '')}

Ingredients:
{chr(10).join(ingredients)}

Produces:
{chr(10).join(results)}
"""

        documents.append({
            "id": recipe_id,
            "type": "recipe",
            "text": text.strip(),
            "metadata": {
                "source": "cdn",
                "table": "recipes"
            }
        })

    return documents


def build_documents(db):
    documents = []

    documents.extend(build_item_documents(db))
    documents.extend(build_recipe_documents(db))
    documents.extend(build_wiki_documents(db))

    for doc in documents:
        doc.setdefault("metadata", {})

        doc["metadata"]["type"] = doc["type"]

        if "name" not in doc["metadata"]:
            lines = doc["text"].splitlines()

            for line in lines:
                if line.startswith("Item: "):
                    doc["metadata"]["name"] = line.replace("Item: ", "")
                    break

                if line.startswith("Recipe: "):
                    doc["metadata"]["name"] = line.replace("Recipe: ", "")
                    break

    return chunk_all_documents(documents)
