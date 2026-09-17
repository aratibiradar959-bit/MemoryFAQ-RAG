import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.services.web_ingestion_service import ingest_web_page
from src.embeddings.embedding_model import get_embedding_model
from src.vectorstore.qdrant_store import get_qdrant_client


URL = "https://docs.python.org/3/tutorial/controlflow.html"


def main():

    print("=" * 60)
    print("WEB INGESTION SERVICE TEST")
    print("=" * 60)

    # ----------------------------------------------
    # Connect to Qdrant
    # ----------------------------------------------

    print("\n1. Connecting to Qdrant...")

    client = get_qdrant_client()

    print("Connected to Qdrant.")

    # ----------------------------------------------
    # Load embedding model
    # ----------------------------------------------

    print("\n2. Loading embedding model...")

    embedding_model = get_embedding_model()

    print("Embedding model loaded.")

    # ----------------------------------------------
    # Ingest webpage
    # ----------------------------------------------

    print("\n3. Ingesting webpage...")

    print(f"URL: {URL}")

    result = ingest_web_page(
        url=URL,
        client=client,
        embedding_model=embedding_model,
    )

    # ----------------------------------------------
    # Display result
    # ----------------------------------------------

    print("\n4. Ingestion result:")

    print(f"Status: {result['status']}")
    print(f"URL: {result['url']}")
    print(f"Chunks: {result['chunks']}")

    print("\n")

    print("=" * 60)
    print("TEST COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    main()