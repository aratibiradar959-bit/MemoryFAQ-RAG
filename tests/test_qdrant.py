from src.vectorstore.qdrant_store import (
    get_qdrant_client,
    create_collection,
    COLLECTION_NAME,
)


def main():
    print("Connecting to Qdrant...")

    client = get_qdrant_client()

    create_collection(client)

    collections = client.get_collections().collections

    print("\nAvailable collections:")

    for collection in collections:
        print("-", collection.name)

    print(f"\nOur collection: {COLLECTION_NAME}")
    print("Qdrant test successful!")


if __name__ == "__main__":
    main()