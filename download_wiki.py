import os
import time
import random
import requests
from mwclient import Site
from mwclient import errors as mw_errors
from config import WIKI_DIR

WIKI_HOST = "wiki.projectgorgon.com"
WIKI_PATH = "/w/"

TARGET_CATEGORIES = ["Game updates", "Game Blogs", "NPCs", "Skills"]

MAX_RETRIES = 5
BASE_DELAY = 0.5
CONNECT_TIMEOUT = 15
READ_TIMEOUT = 60

RESERVED_NAMES = {
    "CON", "NUL", "AUX", "PRN",
    "COM1", "COM2", "COM3", "COM4", "COM5",
    "COM6", "COM7", "COM8", "COM9",
    "LPT1", "LPT2", "LPT3", "LPT4", "LPT5",
    "LPT6", "LPT7", "LPT8", "LPT9",
}

TRANSIENT_API_CODES = {"ratelimited", "maxlag", "readonly"}
MAX_FILENAME_LEN = 150


def get_safe_filename(page_name, index):
    safe_title = "".join(c for c in page_name if c.isalnum() or c in (" ", "_", "-")).rstrip()
    safe_title = safe_title.replace(" ", "_")
    if not safe_title:
        safe_title = f"page_{index}"
    stem = safe_title[:MAX_FILENAME_LEN].rstrip(" .")
    if stem.upper() in RESERVED_NAMES:
        stem += "_"
    return f"{stem}.txt"


def is_transient_error(e):
    if isinstance(e, (requests.exceptions.ConnectionError, requests.exceptions.Timeout)):
        return True
    if isinstance(e, mw_errors.APIError):
        return e.code in TRANSIENT_API_CODES
    return False


def download_page(page, file_path):
    if not page.exists or page.redirect:
        print(f"  Skipping (redirect or deleted): {page.name}")
        return False

    for attempt in range(MAX_RETRIES):
        try:
            content = page.text()
            temp_path = file_path + ".tmp"
            with open(temp_path, "w", encoding="utf-8") as f:
                f.write(content)
            os.replace(temp_path, file_path)
            return True
        except (mw_errors.APIError, requests.exceptions.RequestException) as e:
            if is_transient_error(e) and attempt < MAX_RETRIES - 1:
                delay = (2 ** attempt) + random.uniform(0, 1)
                print(f"  Transient error, retrying in {delay:.1f}s... ({e})")
                time.sleep(delay)
            else:
                print(f"  Failed to download {page.name}: {e}")
                return False
        except Exception as e:
            print(f"  Failed to download {page.name}: {e}")
            return False
    return False


def main():
    WIKI_DIR.mkdir(parents=True, exist_ok=True)

    for f in os.listdir(WIKI_DIR):
        if f.endswith(".tmp"):
            os.remove(os.path.join(WIKI_DIR, f))

    print(f"Connecting to {WIKI_HOST}...")
    site = Site(
        WIKI_HOST,
        path=WIKI_PATH,
        clients_useragent="TwinkleofToesPersonalBackup/1.0 (sabin@figarocastle.org)",
        connection_options={"timeout": (CONNECT_TIMEOUT, READ_TIMEOUT)},
    )

    if TARGET_CATEGORIES:
        print(f"Targeting categories: {', '.join(TARGET_CATEGORIES)}")
        all_pages = []
        for cat_name in TARGET_CATEGORIES:
            try:
                cat = site.categories[cat_name]
                pages = list(cat.members(namespace=0))
                all_pages.extend(pages)
                print(f"  Category '{cat_name}': {len(pages)} pages")
            except mw_errors.APIError as e:
                print(f"  Category '{cat_name}' not found or inaccessible: {e}")
    else:
        print("No category filter set. Fetching all wiki pages...")
        all_pages = site.allpages(namespace=0)

    pages_to_download = list({p.name: p for p in all_pages}.values())
    total_count = len(pages_to_download)
    print(f"Queue verified. Found {total_count} unique items to evaluate.")

    for index, page in enumerate(pages_to_download, 1):
        filename = get_safe_filename(page.name, index)
        file_path = os.path.join(WIKI_DIR, filename)

        if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
            print(f"[{index}/{total_count}] Skipping (Already Downloaded): {page.name}")
            time.sleep(random.uniform(0.1, 0.2))
            continue

        print(f"[{index}/{total_count}] Downloading: {page.name}")
        download_page(page, file_path)
        time.sleep(random.uniform(BASE_DELAY, BASE_DELAY * 2))

    print(f"Complete. Data saved to '{WIKI_DIR}' directory.")


if __name__ == "__main__":
    main()
