from src.embeddings.embedding_model import get_embedding_model
from src.vectorstore.qdrant_store import get_qdrant_client
from src.retrieval.hybrid_retriever import HybridRetriever


def main():

    print("=" * 70)
    print("        HYBRID + RRF + CROSS-ENCODER RETRIEVAL")
    print("=" * 70)

    # --------------------------------------------------
    # Load embedding model
    # --------------------------------------------------

    print("\nLoading embedding model...")

    embedding_model = get_embedding_model()

    # --------------------------------------------------
    # Connect to Qdrant
    # --------------------------------------------------

    print("\nConnecting to Qdrant...")

    client = get_qdrant_client()

    # --------------------------------------------------
    # Create retriever
    # --------------------------------------------------

    print("\nCreating Hybrid Retriever...")

    retriever = HybridRetriever(
        client=client,
        embedding_model=embedding_model,
    )

    # --------------------------------------------------
    # Query
    # --------------------------------------------------

    query = (
         "What are the normal working hours?"
    )

    print("\n" + "=" * 70)
    print("QUERY")
    print("=" * 70)

    print(query)

    # --------------------------------------------------
    # Search
    # --------------------------------------------------

    results = retriever.search(
        query=query,
        top_k=5,
        rerank_top_k=3,
    )

    # ==================================================
    # RRF RESULTS
    # ==================================================

    print("\n" + "=" * 70)
    print("                 RRF RESULTS")
    print("=" * 70)

    for rank, item in enumerate(
        results["rrf_results"],
        start=1,
    ):

        document = item["document"]

        print("\n" + "-" * 70)

        print(
            f"Rank: {rank}"
        )

        print(
            f"RRF Score: "
            f"{item['score']:.6f}"
        )

        print(
            f"Chunk ID: "
            f"{document.metadata.get('chunk_id')}"
        )

        print(
            f"Source: "
            f"{document.metadata.get('original_filename')}"
        )

        print(
            f"Text: "
            f"{document.page_content}"
        )

    # ==================================================
    # CROSS-ENCODER RESULTS
    # ==================================================

    print("\n" + "=" * 70)
    print("             CROSS-ENCODER RERANKING")
    print("=" * 70)

    for rank, item in enumerate(
        results["reranked_results"],
        start=1,
    ):

        document = item["document"]

        print("\n" + "-" * 70)

        print(
            f"Rank: {rank}"
        )

        print(
            f"Cross-Encoder Score: "
            f"{item['score']:.4f}"
        )

        print(
            f"RRF Score: "
            f"{document.metadata.get('rrf_score', 0):.6f}"
        )

        print(
            f"Chunk ID: "
            f"{document.metadata.get('chunk_id')}"
        )

        print(
            f"Source: "
            f"{document.metadata.get('original_filename')}"
        )

        print(
            f"Text: "
            f"{document.page_content}"
        )

    print("\n" + "=" * 70)
    print("Hybrid + RRF + Cross-Encoder test completed.")
    print("=" * 70)


if __name__ == "__main__":
    main()