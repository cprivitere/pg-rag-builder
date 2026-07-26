import hashlib
import json


def document_hash(document):
    content = json.dumps(
        document,
        sort_keys=True,
        ensure_ascii=False
    )

    return hashlib.sha256(
        content.encode("utf-8")
    ).hexdigest()