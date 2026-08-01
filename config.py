from pathlib import Path

PROJECT_ROOT = Path(__file__).parent

DATA_DIR = PROJECT_ROOT / "data"

CDN_DIR = DATA_DIR / "cdn"

WIKI_DIR = DATA_DIR / "wiki"

EMBEDDING_DIM = 512