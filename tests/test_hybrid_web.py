import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.embeddings.embedding_model import get_embedding_model
from src.retrieval.hybrid_retriever import HybridRetriever
from src.vectorstore.qdrant_store import get_qdrant_client


def print_results(title, results):
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)

    for index, document in enumerate(results, start=1):
        print(f"\nResult {index}")
        print("-" * 40)
        print("Source:", document.metadata.get("source"))
        print("File type:", document.metadata.get("file_type"))
        print("URL:", document.metadata.get("url"))
        print("Web hash:", document.metadata.get("web_hash"))
        print("Chunk ID:", document.metadata.get("chunk_id"))
        print("Text:", document.page_content[:300])


def main():

    print("\n" + "=" * 60)
    print("HYBRID WEB + LOCAL RETRIEVAL TEST")
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
    # 3. Create Hybrid Retriever
    # --------------------------------------------------

    print("\nStep 3: Creating Hybrid Retriever...")

    retriever = HybridRetriever(
        client=client,
        embedding_model=embedding_model,
    )

    print("Hybrid Retriever created.")

    # --------------------------------------------------
    # 4. Web question
    # --------------------------------------------------

    web_query = "What is this domain used for?"

    print("\n" + "=" * 60)
    print("WEB QUERY")
    print("=" * 60)

    print("Query:", web_query)

    web_results = retriever.search(
        query=web_query,
        top_k=5,
        rerank_top_k=3,
    )

    print_results(
        "WEB RERANKED RESULTS",
        [
            item["document"]
            for item in web_results["reranked_results"]
        ],
    )

    # --------------------------------------------------
    # 5. Local document question
    # --------------------------------------------------

    local_query = "How many days can employees work remotely?"

    print("\n" + "=" * 60)
    print("LOCAL DOCUMENT QUERY")
    print("=" * 60)

    print("Query:", local_query)

    local_results = retriever.search(
        query=local_query,
        top_k=5,
        rerank_top_k=3,
    )

    print_results(
        "LOCAL RERANKED RESULTS",
        [
            item["document"]
            for item in local_results["reranked_results"]
        ],
    )

    print("\n" + "=" * 60)
    print("TEST COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    main()