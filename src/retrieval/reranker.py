from sentence_transformers import CrossEncoder
from langsmith import traceable


class CrossEncoderReranker:

    def __init__(
        self,
        model_name="cross-encoder/ms-marco-MiniLM-L-6-v2",
    ):

        print("Loading Cross-Encoder model...")

        self.model = CrossEncoder(model_name)

        print("Cross-Encoder loaded successfully.")

    @traceable(name="Relevance Filtering")
    def filter_relevant_documents(
    self,
    documents,
    scores,
    score_threshold: float = -5.0,
):
        results = []

        for document, score in zip(documents, scores):

            score = float(score)

            # Relevance filtering
            if score < score_threshold:
                continue

            results.append(
                {
                    "document": document,
                    "score": score,
                }
            )

        return results

    def rerank(
        self,
        query: str,
        documents,
        top_k: int = 3,
        score_threshold: float = -5.0,
    ):

        # --------------------------------------
        # Create query-document pairs
        # --------------------------------------

        pairs = [
            [query, document.page_content]
            for document in documents
        ]

        # --------------------------------------
        # Get Cross-Encoder scores
        # --------------------------------------

        scores = self.model.predict(pairs)

        # --------------------------------------
        # Relevance filtering
        # --------------------------------------

        results = self.filter_relevant_documents(
            documents=documents,
            scores=scores,
            score_threshold=score_threshold,
        )

        # --------------------------------------
        # Sort by Cross-Encoder score
        # --------------------------------------

        results.sort(
            key=lambda item: item["score"],
            reverse=True,
        )

        # --------------------------------------
        # Return top relevant documents
        # --------------------------------------

        return results[:top_k]