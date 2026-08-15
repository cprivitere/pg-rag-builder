class GameResolver:

    def __init__(self, db):
        self.db = db

    def item(self, item_code):
        return self.db.tables["items"].get(
            f"item_{item_code}"
        )

    def item_name(self, item_code):
        item = self.item(item_code)

        if item:
            return item.get("Name", f"Item {item_code}")

        return f"Unknown Item ({item_code})"

    def ability(self, ability_code):
        return self.db.tables["abilities"].get(
            f"ability_{ability_code}"
        )

    def ability_name(self, ability_code):
        ability = self.ability(ability_code)

        if ability:
            return ability.get("Name", f"Ability {ability_code}")

        return f"Unknown Ability ({ability_code})"

    def recipe(self, recipe_code):
        return self.db.tables["recipes"].get(
            f"recipe_{recipe_code}"
        )

    def recipe_name(self, recipe_code):
        recipe = self.recipe(recipe_code)

        if recipe:
            return recipe.get("Name", f"Recipe {recipe_code}")

        return f"Unknown Recipe ({recipe_code})"