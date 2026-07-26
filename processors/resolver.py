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