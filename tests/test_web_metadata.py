import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.vectorstore.qdrant_store import (
    get_qdrant_client,
    COLLECTION_NAME,
)


def main():
    print("\n" + "=" * 60)
    print("WEB QDRANT METADATA TEST")
    print("=" * 60)

    client = get_qdrant_client()

    points, _ = client.scroll(
        collection_name=COLLECTION_NAME,
        limit=100,
        with_payload=True,
        with_vectors=False,
    )

    for point in points:
        payload = point.payload or {}

        if payload.get("file_type") == "web":
            print("\nWeb document found!")
            print("-" * 60)
            print("Source:", payload.get("source"))
            print("URL:", payload.get("url"))
            print("File type:", payload.get("file_type"))
            print("Web hash:", payload.get("web_hash"))
            print("Chunk ID:", payload.get("chunk_id"))
            print("Text:", payload.get("text"))

    print("\n" + "=" * 60)
    print("TEST COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    main()