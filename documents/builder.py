from processors.resolver import GameResolver


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
                "table": "items"
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
                ingredients.append(
                    f"- {ingredient.get('Desc', 'Unknown ingredient')} "
                    f"(category: {', '.join(ingredient['ItemKeys'])}) "
                    f"x{ingredient.get('StackSize', 1)}"
                )

            # Keyword/group ingredient
            elif "ItemKeys" in ingredient:

                keys = ", ".join(ingredient["ItemKeys"])

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

    return documents