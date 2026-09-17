from pathlib import Path

from langsmith import traceable

from src.memory.conversation_memory import ConversationMemory
from src.chains.query_rewriter import rewrite_query
from src.embeddings.embedding_model import get_embedding_model
from src.vectorstore.qdrant_store import get_qdrant_client
from src.retrieval.hybrid_retriever import HybridRetriever
from src.services.llm import get_llm
from src.prompts.prompts import RAG_PROMPT


class ConversationalRAG:
    """
    Conversational RAG pipeline.

    Pipeline:

        User Question
              ↓
        Conversation Memory
              ↓
        Query Rewriting
              ↓
        Hybrid Retrieval
        ┌─────┴─────┐
        ↓           ↓
      Dense        BM25
        └─────┬─────┘
              ↓
          RRF Fusion
              ↓
      Cross-Encoder Reranking
              ↓
      Relevance Filtering
              ↓
          Context
              ↓
             LLM
              ↓
        Answer + Sources

    Supports:

        - Local documents
        - Web pages
        - Conversational memory
        - Query rewriting
        - Dense retrieval
        - BM25 retrieval
        - RRF fusion
        - Cross-Encoder reranking
        - Relevance filtering
    """

    def __init__(self):

        print("Initializing Conversational RAG...")

        # ==================================================
        # 1. Conversation Memory
        # ==================================================

        self.memory = ConversationMemory()

        # ==================================================
        # 2. Embedding Model
        # ==================================================

        print("Loading embedding model...")

        self.embedding_model = get_embedding_model()

        print("Embedding model loaded.")

        # ==================================================
        # 3. Qdrant Client
        # ==================================================

        print("Connecting to Qdrant...")

        self.client = get_qdrant_client()

        print("Connected to Qdrant.")

        # ==================================================
        # 4. Hybrid Retriever
        # ==================================================

        print("Initializing Hybrid Retriever...")

        self.retriever = HybridRetriever(
            client=self.client,
            embedding_model=self.embedding_model,
        )

        print("Hybrid Retriever initialized.")

        # ==================================================
        # 5. LLM
        # ==================================================

        print("Loading LLM...")

        self.llm = get_llm()

        print("LLM loaded.")

        print(
            "\nConversational RAG initialized successfully."
        )

    # ======================================================
    # REFRESH RETRIEVER
    # ======================================================

    def refresh_retriever(self):
        """
        Recreate the Hybrid Retriever so that
        newly ingested documents are available
        to BM25 and dense retrieval.
        """

        print("\nRefreshing Hybrid Retriever...")

        self.retriever = HybridRetriever(
            client=self.client,
            embedding_model=self.embedding_model,
        )

        print(
            "Hybrid Retriever refreshed successfully."
        )

    # ======================================================
    # SOURCE HELPER
    # ======================================================

    def get_source(self, document):
        """
        Get a clean source name.

        Web document:
            returns URL

        Local document:
            returns filename
        """

        file_type = document.metadata.get(
            "file_type"
        )

        # --------------------------------------------------
        # Web source
        # --------------------------------------------------

        if file_type == "web":

            return (
                document.metadata.get("url")
                or document.metadata.get("source")
                or "Unknown"
            )

        # --------------------------------------------------
        # Local document source
        # --------------------------------------------------

        source = (
            document.metadata.get(
                "original_filename"
            )
            or document.metadata.get(
                "source"
            )
            or "Unknown"
        )

        return Path(source).name

    # ======================================================
    # ASK
    # ======================================================

    @traceable(name="Conversational RAG")
    def ask(self, question: str):

        # ==================================================
        # STEP 1: Get Conversation History
        # ==================================================

        history = self.memory.get_history()

        # ==================================================
        # STEP 2: Rewrite User Question
        # ==================================================

        standalone_question = rewrite_query(
            question=question,
            conversation_history=history,
        )

        print("\n" + "=" * 60)
        print("QUESTION PROCESSING")
        print("=" * 60)

        print("\nOriginal question:")
        print(question)

        print("\nStandalone question:")
        print(standalone_question)

        # ==================================================
        # STEP 3: Hybrid Retrieval
        # ==================================================

        print("\n" + "=" * 60)
        print("HYBRID RETRIEVAL")
        print("=" * 60)

        retrieval_results = self.retriever.search(
            query=standalone_question,
            top_k=5,
            rerank_top_k=3,
            score_threshold=-5.0,
        )

        # ==================================================
        # STEP 4: Get Cross-Encoder Results
        # ==================================================

        reranked_results = retrieval_results[
            "reranked_results"
        ]

        print(
            f"\nRelevant documents found: "
            f"{len(reranked_results)}"
        )

        # ==================================================
        # STEP 5: Handle No Relevant Results
        # ==================================================

        if not reranked_results:

            answer = (
                "I couldn't find relevant information "
                "in the provided documents or web pages."
            )

            # Save conversation

            self.memory.add_user_message(
                question
            )

            self.memory.add_ai_message(
                answer
            )

            return {
                "question": question,
                "standalone_question": standalone_question,
                "answer": answer,
                "sources": [],
            }

        # ==================================================
        # STEP 6: Build Context
        # ==================================================

        context_parts = []

        print("\n" + "=" * 60)
        print("RETRIEVED CONTEXT")
        print("=" * 60)

        for item in reranked_results:

            document = item["document"]

            score = item["score"]

            # Get source

            source = self.get_source(
                document
            )

            text = document.page_content

            file_type = document.metadata.get(
                "file_type"
            )

            print("\nRetrieved document:")
            print(f"Source: {source}")
            print(f"Type: {file_type}")
            print(
                f"Cross-Encoder Score: "
                f"{score:.4f}"
            )

            # --------------------------------------------------
            # Add source + text to context
            # --------------------------------------------------

            context_parts.append(
                f"Source: {source}\n"
                f"Content:\n{text}"
            )

        context = "\n\n".join(
            context_parts
        )

        # ==================================================
        # STEP 7: Create RAG Prompt
        # ==================================================

        messages = RAG_PROMPT.format_messages(
            context=context,
            question=standalone_question,
        )

        # ==================================================
        # STEP 8: Call LLM
        # ==================================================

        print("\n" + "=" * 60)
        print("CALLING LLM")
        print("=" * 60)

        response = self.llm.invoke(
            messages
        )

        answer = response.content

        # ==================================================
        # STEP 9: Save Conversation
        # ==================================================

        self.memory.add_user_message(
            question
        )

        self.memory.add_ai_message(
            answer
        )

        # ==================================================
        # STEP 10: Prepare Sources
        # ==================================================

        sources = []

        for item in reranked_results:

            document = item["document"]

            source_name = self.get_source(
                document
            )

            if source_name not in sources:

                sources.append(
                    source_name
                )

        # ==================================================
        # STEP 11: Return Final Result
        # ==================================================

        return {
            "question": question,
            "standalone_question": standalone_question,
            "answer": answer,
            "sources": sources,
        }