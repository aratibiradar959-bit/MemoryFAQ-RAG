from src.embeddings.embedding_model import get_embedding_model
from src.vectorstore.qdrant_store import get_qdrant_client
from src.retrieval.hybrid_retriever import HybridRetriever


QUESTIONS = [
    "How many days can employees work remotely?",
    "What are the normal working hours?",
    "What is Apple's stock price?",
    "What is the weather today?",
    "Who is the CEO of Google?",
]


def main():

    print("=" * 70)
    print("CROSS-ENCODER SCORE INSPECTION")
    print("=" * 70)

    embedding_model = get_embedding_model()

    client = get_qdrant_client()

    retriever = HybridRetriever(
        client=client,
        embedding_model=embedding_model,
    )

    for question in QUESTIONS:

        print("\n" + "=" * 70)
        print("QUESTION")
        print("=" * 70)

        print(question)

        results = retriever.search(
            query=question,
            top_k=5,
            rerank_top_k=5,
            score_threshold=-100.0,
        )

        print("\nCross-Encoder Scores:")
        print("-" * 70)

        for result in results["reranked_results"]:

            document = result["document"]
            score = result["score"]

            source = (
                document.metadata.get("original_filename")
                or document.metadata.get("source")
            )

            print(
                f"{source:<30} Score: {score:.4f}"
            )


if __name__ == "__main__":
    main()