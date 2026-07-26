from dataclasses import dataclass, field


@dataclass
class GameDatabase:

    tables: dict = field(default_factory=dict)

    wiki: dict = field(default_factory=dict)