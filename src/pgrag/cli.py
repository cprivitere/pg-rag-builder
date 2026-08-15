import argparse
import sys


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="pgrag",
        description="Project Gorgon RAG pipeline: fetch data, build documents and index.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("download-cdn", help="Fetch CDN JSON exports to data/cdn/")
    p.set_defaults(func=_download_cdn)

    p = sub.add_parser("download-wiki", help="Fetch wiki pages to data/wiki/ (slow)")
    p.set_defaults(func=_download_wiki)

    p = sub.add_parser("build-documents", help="Generate data/documents.json from CDN + wiki")
    p.set_defaults(func=_build_documents)

    p = sub.add_parser("build-index", help="Upsert documents into the ChromaDB index")
    p.set_defaults(func=_build_index)

    p = sub.add_parser("validate", help="Validate vectorstore health (exit 1 if issues)")
    p.set_defaults(func=_validate)

    args = parser.parse_args()
    rc = args.func()
    sys.exit(rc if isinstance(rc, int) else 0)


def _download_cdn() -> int:
    from pgrag.loaders.download_cdn import download_cdn

    download_cdn()
    return 0


def _download_wiki() -> int:
    from pgrag.loaders.download_wiki import main as sync_wiki

    return sync_wiki()


def _build_documents() -> int:
    from pgrag.build import generate_documents

    generate_documents()
    return 0


def _build_index() -> int:
    from pgrag.vectorstore.build_index import build_index

    build_index()
    return 0


def _validate() -> int:
    from pgrag.vectorstore.health_check import health_check

    return health_check()


if __name__ == "__main__":
    main()