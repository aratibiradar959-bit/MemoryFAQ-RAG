import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


from src.loaders.web_loader import load_web_page
from src.retrieval.text_splitter import split_documents
from src.embeddings.embedding_model import get_embedding_model

from src.vectorstore.qdrant_store import (
    get_qdrant_client,
    create_collection,
    add_documents,
)


URL = "https://example.com"


def main():

    print("\n" + "=" * 60)
    print("WEB → QDRANT TEST")
    print("=" * 60)

    # --------------------------------------------------
    # 1. Load webpage
    # --------------------------------------------------

    print("\nStep 1: Loading webpage...")

    documents = load_web_page(URL)

    print(f"Documents loaded: {len(documents)}")

    # --------------------------------------------------
    # 2. Split webpage
    # --------------------------------------------------

    print("\nStep 2: Splitting webpage...")

    chunks = split_documents(documents)

    print(f"Chunks created: {len(chunks)}")

    # --------------------------------------------------
    # 3. Load embedding model
    # --------------------------------------------------

    print("\nStep 3: Loading embedding model...")

    embedding_model = get_embedding_model()

    print("Embedding model loaded.")

    # --------------------------------------------------
    # 4. Connect to Qdrant
    # --------------------------------------------------

    print("\nStep 4: Connecting to Qdrant...")

    client = get_qdrant_client()

    create_collection(client)

    print("Connected to Qdrant.")

    # --------------------------------------------------
    # 5. Store web chunks
    # --------------------------------------------------

    print("\nStep 5: Storing web chunks...")

    add_documents(
        client,
        chunks,
        embedding_model,
    )

    print("\nWeb page stored successfully!")

    # --------------------------------------------------
    # 6. Display metadata
    # --------------------------------------------------

    print("\nStored chunk metadata:")

    for index, chunk in enumerate(chunks, start=1):

        print(f"\nChunk {index}")
        print(chunk.metadata)

    print("\n" + "=" * 60)
    print("WEB → QDRANT TEST COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    main()