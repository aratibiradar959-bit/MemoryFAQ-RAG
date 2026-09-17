import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


from src.embeddings.embedding_model import get_embedding_model
from src.vectorstore.qdrant_store import get_qdrant_client
from src.retrieval.retriever import search_documents


QUERY = "What is this domain used for?"


def main():

    print("\n" + "=" * 60)
    print("WEB RETRIEVAL TEST")
    print("=" * 60)

    # --------------------------------------------------
    # 1. Load embedding model
    # --------------------------------------------------

    print("\nStep 1: Loading embedding model...")

    embedding_model = get_embedding_model()

    print("Embedding model loaded.")

    # --------------------------------------------------
    # 2. Connect to Qdrant
    # --------------------------------------------------

    print("\nStep 2: Connecting to Qdrant...")

    client = get_qdrant_client()

    print("Connected to Qdrant.")

    # --------------------------------------------------
    # 3. Search
    # --------------------------------------------------

    print(f"\nStep 3: Searching for:")
    print(f"Query: {QUERY}")

    results = search_documents(
        client=client,
        embedding_model=embedding_model,
        query=QUERY,
        top_k=5,
        score_threshold=0.0,
    )

    # --------------------------------------------------
    # 4. Display results
    # --------------------------------------------------

    print(f"\nDocuments retrieved: {len(results)}")

    for index, result in enumerate(results, start=1):

        payload = result.payload or {}

        print("\n" + "-" * 60)
        print(f"RESULT {index}")
        print("-" * 60)

        print(f"Score: {result.score}")
        print(f"Source: {payload.get('source')}")
        print(f"File Type: {payload.get('file_type')}")
        print(f"URL: {payload.get('url')}")
        print(f"\nText:\n{payload.get('text')}")

    print("\n" + "=" * 60)
    print("WEB RETRIEVAL TEST COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    main()