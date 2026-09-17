# MemoryFAQ-RAG

A production-oriented conversational RAG chatbot that answers questions from local documents and web pages using hybrid retrieval, RRF fusion, Cross-Encoder reranking, conversation memory, and query rewriting.

## 🚀 Project Overview

MemoryFAQ-RAG is a multi-document conversational Question Answering system built using Retrieval-Augmented Generation (RAG).

The system retrieves relevant information from:

- Local documents
- Web pages

It then uses an LLM to generate grounded answers based on the retrieved context.

The chatbot also maintains conversation history, allowing users to ask follow-up questions naturally.

## ✨ Key Features

- Multi-document RAG
- TXT, PDF, CSV, DOCX, and XLSX document support
- Qdrant vector database
- Hugging Face embeddings
- Dense vector retrieval
- BM25 sparse retrieval
- Hybrid search
- Reciprocal Rank Fusion (RRF)
- Cross-Encoder reranking
- Cross-Encoder relevance filtering
- Similarity threshold filtering
- Conversational memory
- Query rewriting
- Web page ingestion
- Local documents + web content retrieval
- Source citations
- Dynamic document upload
- Document listing and deletion
- Unsupported-question handling
- LangSmith tracing and monitoring
- Streamlit user interface
- Docker containerization
- RAG evaluation

## 🏗️ Architecture

```text
                         User
                           |
                           v
                    Streamlit UI
                           |
                           v
                 Conversational RAG
                           |
                    Query Rewriting
                           |
                           v
                  Hybrid Retrieval
                    /           \
                   /             \
                  v               v
          Dense Retrieval      BM25 Search
             (Qdrant)          (Sparse)
                  \               /
                   \             /
                    v           v
                  RRF Fusion
                       |
                       v
              Cross-Encoder Reranking
                       |
                       v
             Relevance Filtering
                       |
                       v
                  Top Context
                       |
                       v
                      LLM
                       |
                       v
                  Final Answer
                       |
                       v
                Source Citations