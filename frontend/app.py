import sys
from pathlib import Path

import streamlit as st

# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# IMPORTS
# ============================================================

from src.services.chatbot_service import ChatbotService
from src.services.document_ingestion_service import (
    ingest_uploaded_files,
)
from src.services.web_ingestion_service import (
    ingest_web_page,
)
from src.vectorstore.qdrant_store import (
    get_all_documents,
    delete_document,
    delete_web_document,
)


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="MemoryFAQ-RAG",
    page_icon="🧠",
    layout="wide",
)


# ============================================================
# TITLE
# ============================================================

st.title("🧠 MemoryFAQ-RAG")

st.caption(
    "Multi-Turn FAQ Chatbot powered by "
    "RAG + Qdrant + Hybrid Search + OpenRouter"
)


# ============================================================
# INITIALIZE CHATBOT
# ============================================================

if "chatbot" not in st.session_state:
    st.session_state.chatbot = ChatbotService()


chatbot = st.session_state.chatbot


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("⚙️ Chat Settings")

    # ========================================================
    # SYSTEM STATUS
    # ========================================================

    with st.expander("🟢 System Status", expanded=True):
        st.success("Qdrant: Connected")
        st.success("RAG Pipeline: Ready")
        st.success("Web RAG: Enabled")
        st.success("Hybrid Search: Enabled")

    # ========================================================
    # LOCAL DOCUMENTS
    # ========================================================

    with st.expander("📄 Local Documents", expanded=False):

        uploaded_files = st.file_uploader(
            "Choose documents",
            type=[
                "txt",
                "pdf",
                "docx",
                "csv",
                "xlsx",
            ],
            accept_multiple_files=True,
        )

        if uploaded_files:
            st.caption(f"Selected {len(uploaded_files)} document(s)")

            if st.button(
                "🚀 Ingest Documents",
                use_container_width=True,
            ):
                with st.spinner("Processing documents..."):
                    try:
                        result = ingest_uploaded_files(
                            uploaded_files,
                            chatbot.chatbot.client,
                            chatbot.chatbot.embedding_model,
                        )

                        if result["uploaded"]:
                            st.success(
                                f"Successfully ingested "
                                f"{result['chunks']} chunks."
                            )

                            for filename in result["uploaded"]:
                                st.write(f"✅ Added: {filename}")

                            chatbot.chatbot.refresh_retriever()
                            st.success("🔄 Retriever refreshed successfully.")

                        if result["duplicates"]:
                            for filename in result["duplicates"]:
                                st.warning(
                                    f"⚠️ Duplicate file skipped: {filename}"
                                )

                    except Exception as e:
                        st.error(f"Error during ingestion: {e}")

    # ========================================================
    # WEB PAGES
    # ========================================================

    with st.expander("🌐 Web Pages", expanded=False):

        web_url = st.text_input(
            "Enter webpage URL",
            placeholder="https://example.com",
        )

        if st.button(
            "🚀 Ingest Web Page",
            use_container_width=True,
        ):
            if not web_url.strip():
                st.warning("Please enter a webpage URL.")
            else:
                with st.spinner("Loading and processing webpage..."):
                    try:
                        result = ingest_web_page(
                            url=web_url,
                            client=chatbot.chatbot.client,
                            embedding_model=chatbot.chatbot.embedding_model,
                        )

                        if result["status"] == "success":
                            st.success("Webpage ingested successfully!")
                            st.caption(f"📦 Chunks stored: {result['chunks']}")

                            chatbot.chatbot.refresh_retriever()
                            st.success("🔄 Retriever refreshed successfully.")

                        elif result["status"] == "duplicate":
                            st.warning(
                                "⚠️ This webpage is already stored "
                                "in the knowledge base."
                            )

                    except Exception as e:
                        st.error(f"Error during web ingestion: {e}")

    # ========================================================
    # KNOWLEDGE BASE
    # ========================================================

    with st.expander("📚 Knowledge Base", expanded=False):

        try:
            documents = get_all_documents(chatbot.chatbot.client)

            if not documents:
                st.info("No documents or web pages found.")

            else:
                local_documents = [
                    document
                    for document in documents
                    if document["file_type"] != "web"
                ]

                web_documents = [
                    document
                    for document in documents
                    if document["file_type"] == "web"
                ]

                if local_documents:
                    st.markdown("**📄 Local Documents**")

                    for document in local_documents:
                        filename = document["filename"]
                        file_type = document["file_type"]
                        chunk_count = document["chunk_count"]
                        file_hash = document.get("file_hash")

                        st.write(f"📄 **{filename}**")
                        st.caption(
                            f"Type: {file_type.upper()} | "
                            f"Chunks: {chunk_count}"
                        )

                        if st.button(
                            f"🗑️ Delete {filename}",
                            key=f"delete_file_{file_hash}",
                            use_container_width=True,
                        ):
                            try:
                                delete_document(
                                    chatbot.chatbot.client,
                                    file_hash,
                                )
                                chatbot.chatbot.refresh_retriever()
                                st.success(f"{filename} deleted.")
                                st.rerun()
                            except Exception as e:
                                st.error(
                                    f"Error deleting {filename}: {e}"
                                )

                if web_documents:
                    st.markdown("**🌐 Web Pages**")

                    for document in web_documents:
                        url = document.get("url") or document["filename"]
                        web_hash = document.get("web_hash")
                        chunk_count = document["chunk_count"]

                        st.write(f"🌐 **{url}**")
                        st.caption(
                            f"Type: WEB | Chunks: {chunk_count}"
                        )

                        if st.button(
                            "🗑️ Delete Web Page",
                            key=f"delete_web_{web_hash}",
                            use_container_width=True,
                        ):
                            try:
                                delete_web_document(
                                    chatbot.chatbot.client,
                                    web_hash,
                                )
                                chatbot.chatbot.refresh_retriever()
                                st.success("Web page deleted.")
                                st.rerun()
                            except Exception as e:
                                st.error(
                                    f"Error deleting web page: {e}"
                                )

        except Exception as e:
            st.error(f"Could not load documents: {e}")

    # ========================================================
    # CONVERSATION
    # ========================================================

    with st.expander("💬 Conversation", expanded=False):

        if st.button(
            "🗑️ Clear Conversation",
            use_container_width=True,
        ):
            chatbot.clear_memory()
            st.rerun()


# ============================================================
# WELCOME SCREEN
# ============================================================

history = chatbot.get_history()

if not history:

    with st.container(border=True):

        st.markdown("## 💬 Start a conversation")

        st.write(
            "Ask questions about your uploaded documents, "
            "company policies, or ingested web pages."
        )

        st.write(
            "The chatbot uses hybrid retrieval, reranking, "
            "and conversational memory to find relevant information."
        )

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.info("🔍 Hybrid Search")

        with col2:
            st.info("🌐 Web RAG")

        with col3:
            st.info("🧠 Memory")

        with col4:
            st.info("📚 Source Citations")


# ============================================================
# CHAT HISTORY
# ============================================================

for message in history:

    if message["role"] == "user":

        with st.chat_message("user", avatar="👤"):
            st.caption("You")
            st.markdown(message["content"])

    else:

        with st.chat_message("assistant", avatar="🤖"):
            st.caption("MemoryFAQ-RAG")
            st.markdown(message["content"])


# ============================================================
# CHAT INPUT
# ============================================================

question = st.chat_input(
    "Ask a question about your documents or web pages..."
)


if question:

    # ========================================================
    # USER MESSAGE
    # ========================================================

    with st.chat_message("user", avatar="👤"):
        st.caption("You")
        st.markdown(question)

    # ========================================================
    # ASSISTANT RESPONSE
    # ========================================================

    with st.chat_message("assistant", avatar="🤖"):
        st.caption("MemoryFAQ-RAG")

        with st.spinner(
            "Searching documents and web pages..."
        ):

            result = chatbot.ask(
                question
            )

        # ----------------------------------------------------
        # Answer
        # ----------------------------------------------------

        st.markdown(result["answer"])

        # ----------------------------------------------------
        # Sources
        # ----------------------------------------------------

    if result["sources"]:

        with st.expander("📚 Sources"):
    
            for source in result["sources"]:
    
                # Web source
                if source.startswith("http://") or source.startswith("https://"):
    
                    st.markdown(
                        f"🌐 **Web Source**  \n"
                        f"[Open webpage]({source})"
                    )
    
                # Local document
                else:
    
                    st.markdown(
                        f"📄 **Local Document:** `{source}`"
                    )