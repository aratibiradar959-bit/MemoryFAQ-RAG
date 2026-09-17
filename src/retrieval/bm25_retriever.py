import re

from rank_bm25 import BM25Okapi


class BM25Retriever:
    """
    BM25 sparse retriever.

    It searches documents using keyword matching.
    """

    STOP_WORDS = {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "for",
        "from",
        "has",
        "have",
        "how",
        "i",
        "in",
        "is",
        "it",
        "of",
        "on",
        "or",
        "that",
        "the",
        "this",
        "to",
        "was",
        "what",
        "when",
        "where",
        "which",
        "who",
        "why",
        "with",
    }

    def __init__(self, documents):
        """
        Initialize BM25 with documents.

        Args:
            documents: List of LangChain Document objects.
        """

        self.documents = documents

        # Extract text from each document
        self.texts = [
            document.page_content
            for document in documents
        ]

        # Tokenize documents
        tokenized_documents = [
            self.tokenize(text)
            for text in self.texts
        ]

        # Create BM25 index
        self.bm25 = BM25Okapi(
            tokenized_documents
        )

    def tokenize(self, text):
        """
        Convert text into lowercase meaningful words.

        Common stop words are removed so that BM25
        focuses on useful keywords.
        """

        text = text.lower()

        # Keep only words and numbers
        words = re.findall(
            r"\b[a-z0-9]+\b",
            text,
        )

        # Remove common stop words
        words = [
            word
            for word in words
            if word not in self.STOP_WORDS
        ]

        return words

    def search(
        self,
        query: str,
        top_k: int = 5,
    ):
        """
        Search documents using BM25.

        Documents with a BM25 score of 0 are ignored.

        Returns:
            List of tuples:
            (document, score)
        """

        # Tokenize query
        tokenized_query = self.tokenize(
            query
        )

        # Calculate BM25 scores
        scores = self.bm25.get_scores(
            tokenized_query
        )

        # Sort document indexes by score
        ranked_indexes = sorted(
            range(len(scores)),
            key=lambda index: scores[index],
            reverse=True,
        )

        results = []

        for index in ranked_indexes:

            score = float(scores[index])

            # Ignore documents with no keyword relevance
            if score <= 0:
                continue

            results.append(
                (
                    self.documents[index],
                    score,
                )
            )

            # Stop after collecting top_k relevant documents
            if len(results) >= top_k:
                break

        return results