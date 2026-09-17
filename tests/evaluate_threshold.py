from src.embeddings.embedding_model import get_embedding_model
from src.vectorstore.qdrant_store import get_qdrant_client
from src.retrieval.hybrid_retriever import HybridRetriever

from tests.evaluation_dataset import EVALUATION_DATASET


def evaluate_threshold(
    retriever,
    threshold,
):
    """
    Evaluate one Cross-Encoder threshold.

    Supported questions:
        Calculate Precision, Recall and F1.

    Unsupported questions:
        Calculate Unsupported Rejection Rate.
    """

    # ==========================================
    # SUPPORTED QUESTION METRICS
    # ==========================================

    true_positive = 0
    false_positive = 0
    false_negative = 0

    # ==========================================
    # UNSUPPORTED QUESTION METRICS
    # ==========================================

    unsupported_total = 0
    unsupported_rejected = 0

    # ==========================================
    # EVALUATE EACH QUESTION
    # ==========================================

    for item in EVALUATION_DATASET:

        question = item["question"]

        question_type = item["type"]

        expected_sources = set(
            item["relevant_sources"]
        )

        # ------------------------------------------
        # Run retrieval
        # ------------------------------------------

        results = retriever.search(
            query=question,
            top_k=5,
            rerank_top_k=5,
            score_threshold=threshold,
        )

        retrieved_sources = set()

        for result in results["reranked_results"]:

            document = result["document"]

            source = (
                document.metadata.get(
                    "original_filename"
                )
                or document.metadata.get(
                    "source"
                )
            )

            if source:
                retrieved_sources.add(source)

        # ==========================================
        # SUPPORTED QUESTION
        # ==========================================

        if question_type == "supported":

            true_positive += len(
                retrieved_sources & expected_sources
            )

            false_positive += len(
                retrieved_sources - expected_sources
            )

            false_negative += len(
                expected_sources - retrieved_sources
            )

        # ==========================================
        # UNSUPPORTED QUESTION
        # ==========================================

        elif question_type == "unsupported":

            unsupported_total += 1

            # No document should be returned.
            if len(retrieved_sources) == 0:

                unsupported_rejected += 1

    # ==========================================
    # PRECISION
    # ==========================================

    precision = (
        true_positive
        / (true_positive + false_positive)
        if (true_positive + false_positive) > 0
        else 0
    )

    # ==========================================
    # RECALL
    # ==========================================

    recall = (
        true_positive
        / (true_positive + false_negative)
        if (true_positive + false_negative) > 0
        else 0
    )

    # ==========================================
    # F1 SCORE
    # ==========================================

    f1 = (
        2 * (precision * recall)
        / (precision + recall)
        if (precision + recall) > 0
        else 0
    )

    # ==========================================
    # UNSUPPORTED REJECTION RATE
    # ==========================================

    unsupported_rejection_rate = (
        unsupported_rejected
        / unsupported_total
        if unsupported_total > 0
        else 0
    )

    return (
        precision,
        recall,
        f1,
        unsupported_rejection_rate,
    )


def main():

    print("=" * 80)
    print("CROSS-ENCODER THRESHOLD EVALUATION")
    print("=" * 80)

    # ==========================================
    # LOAD MODELS ONCE
    # ==========================================

    print("\nLoading embedding model...")

    embedding_model = get_embedding_model()

    print("\nConnecting to Qdrant...")

    client = get_qdrant_client()

    print("\nCreating Hybrid Retriever...")

    retriever = HybridRetriever(
        client=client,
        embedding_model=embedding_model,
    )

    # ==========================================
    # THRESHOLDS TO TEST
    # ==========================================

    thresholds = [
        -5.0,
        -4.0,
        -3.0,
        -2.0,
        -1.0,
        0.0,
        1.0,
    ]

    # ==========================================
    # PRINT HEADER
    # ==========================================

    print("\n")
    print("=" * 80)
    print("RESULTS")
    print("=" * 80)

    print(
        f"{'Threshold':<12}"
        f"{'Precision':<15}"
        f"{'Recall':<15}"
        f"{'F1 Score':<15}"
        f"{'Unsupported Reject':<20}"
    )

    print("-" * 80)

    # ==========================================
    # TEST EACH THRESHOLD
    # ==========================================

    for threshold in thresholds:

        (
            precision,
            recall,
            f1,
            unsupported_rejection_rate,
        ) = evaluate_threshold(
            retriever=retriever,
            threshold=threshold,
        )

        print(
            f"{threshold:<12.1f}"
            f"{precision:<15.2%}"
            f"{recall:<15.2%}"
            f"{f1:<15.2%}"
            f"{unsupported_rejection_rate:<20.2%}"
        )


if __name__ == "__main__":
    main()