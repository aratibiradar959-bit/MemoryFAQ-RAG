from langchain_core.documents import Document
from langsmith import traceable

from src.retrieval.retriever import search_documents
from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.rrf import reciprocal_rank_fusion
from src.retrieval.reranker import CrossEncoderReranker

from src.vectorstore.qdrant_store import get_all_chunks


class HybridRetriever:

    def __init__(self, client, embedding_model):

        self.client = client
        self.embedding_model = embedding_model

        # Load all documents for BM25
        documents = get_all_chunks(client)

        self.bm25 = BM25Retriever(documents)

        # Cross-Encoder reranker
        self.reranker = CrossEncoderReranker()

    def refresh_retriever(self):

        """
        Reload documents after new documents/web pages
        are added to Qdrant.
        """

        documents = get_all_chunks(self.client)

        self.bm25 = BM25Retriever(documents)

    @traceable(name="Dense Search")
    def dense_search(self, query: str, top_k: int = 5):

        results = search_documents(
            client=self.client,
            embedding_model=self.embedding_model,
            query=query,
            top_k=top_k,
        )

        documents = []

        for result in results:

            payload = result.payload or {}

            document = Document(
                page_content=payload.get("text", ""),
                metadata={
                    "source": payload.get("source"),
                    "file_type": payload.get("file_type"),
                    "url": payload.get("url"),
                    "web_hash": payload.get("web_hash"),
                    "file_hash": payload.get("file_hash"),
                    "original_filename": payload.get("original_filename"),
                    "chunk_id": payload.get("chunk_id"),
                    "dense_score": result.score,
                },
            )

            documents.append(document)

        return documents

    @traceable(name="BM25 Search")
    def sparse_search(self, query: str, top_k: int = 5):

        results = self.bm25.search(
            query=query,
            top_k=top_k,
        )

        documents = []

        for document, score in results:

            document.metadata["bm25_score"] = float(score)

            documents.append(document)

        return documents

    @traceable(name="RRF Fusion")
    def fuse_results(
        self,
        dense_results,
        sparse_results,
    ):

        return reciprocal_rank_fusion(
            [
                dense_results,
                sparse_results,
            ]
        )

    @traceable(name="Cross-Encoder Reranking")
    def cross_encoder_rerank(
        self,
        query: str,
        documents,
        top_k: int = 3,
        score_threshold: float = -5.0,
    ):

        return self.reranker.rerank(
            query=query,
            documents=documents,
            top_k=top_k,
            score_threshold=score_threshold,
        )

    @traceable(name="Hybrid Retrieval")
    def search(
        self,
        query: str,
        top_k: int = 5,
        rerank_top_k: int = 3,
        score_threshold: float = -5.0,
    ):

        # --------------------------------------
        # Dense Search
        # --------------------------------------

        dense_results = self.dense_search(
            query=query,
            top_k=top_k,
        )

        # --------------------------------------
        # BM25 Search
        # --------------------------------------

        sparse_results = self.sparse_search(
            query=query,
            top_k=top_k,
        )

        # --------------------------------------
        # RRF Fusion
        # --------------------------------------

        rrf_results = self.fuse_results(
            dense_results=dense_results,
            sparse_results=sparse_results,
        )

        rrf_results = rrf_results[:top_k]

        rrf_documents = []

        for item in rrf_results:

            document = item["document"]

            document.metadata["rrf_score"] = item["score"]

            rrf_documents.append(document)

        # --------------------------------------
        # Cross-Encoder Reranking
        # --------------------------------------

        reranked_results = self.cross_encoder_rerank(
            query=query,
            documents=rrf_documents,
            top_k=rerank_top_k,
            score_threshold=score_threshold,
        )

        # --------------------------------------
        # Return all retrieval stages
        # --------------------------------------

        return {
            "dense_results": dense_results,
            "sparse_results": sparse_results,
            "rrf_results": rrf_results,
            "reranked_results": reranked_results,
        }