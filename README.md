# MemoryFAQ-RAG

A conversational Retrieval-Augmented Generation (RAG) application for answering questions from local documents and web pages. It combines dense retrieval, BM25, Reciprocal Rank Fusion (RRF), reranking, query rewriting, and conversation memory to produce grounded answers with source citations.

## Features

- Chat with local `.txt`, `.pdf`, `.csv`, `.docx`, and `.xlsx` files
- Ingest a public web page directly from the Streamlit interface
- Hybrid retrieval: semantic vector search plus BM25 keyword search
- RRF fusion and Cross-Encoder reranking for stronger result quality
- Relevance and similarity-threshold filtering to reduce unsupported answers
- Follow-up question support through conversation memory and query rewriting
- Source citations for local documents and web pages
- Duplicate detection and document deletion
- Local Qdrant vector storage, Docker support, and evaluation tests

## Architecture

```text
User → Streamlit UI → Conversational RAG → Query Rewriter
                                             ↓
                                    Hybrid Retrieval
                                    ↙                ↘
                          Dense search (Qdrant)     BM25 search
                                    ↘                ↙
                                      RRF fusion
                                           ↓
                                Cross-Encoder reranker
                                           ↓
                                  Relevance filtering
                                           ↓
                              LLM answer + citations
```

## Tech stack

| Area | Tools |
| --- | --- |
| Application | Python, Streamlit |
| Orchestration | LangChain, LangGraph |
| LLM | OpenRouter |
| Embeddings | Hugging Face Sentence Transformers |
| Vector database | Qdrant (local mode) |
| Retrieval | Dense search, BM25, RRF, Cross-Encoder |
| Observability | LangSmith (optional) |

## Quick start

### Prerequisites

- Python 3.12 or later
- An [OpenRouter API key](https://openrouter.ai/keys)

### 1. Clone and install

```bash
git clone https://github.com/aratibiradar959-bit/MemoryFAQ-RAG.git
cd MemoryFAQ-RAG
python -m venv .venv
```

Activate the environment:

```bash
# Windows PowerShell
.venv\Scripts\Activate.ps1

# macOS/Linux
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

### 2. Configure environment variables

Copy the example file and add your API key:

```bash
copy .env.example .env   # Windows
# cp .env.example .env   # macOS/Linux
```

```env
OPENROUTER_API_KEY=your_openrouter_api_key

# Optional LangSmith tracing
LANGSMITH_TRACING=false
LANGSMITH_API_KEY=
LANGSMITH_PROJECT=MemoryFAQ-RAG
```

Never commit `.env`; it is already excluded by `.gitignore`.

### 3. Add knowledge and run the app

Place sample files in `data/`, then ingest them:

```bash
python scripts/ingest.py
```

Start the application:

```bash
streamlit run frontend/app.py
```

Open the local URL displayed by Streamlit. You can upload files or ingest a web page from the sidebar.

## Docker

Build and run the app:

```bash
docker build -t memoryfaq-rag .
docker run --rm -p 8501:8501 --env-file .env memoryfaq-rag
```

Then visit `http://localhost:8501`.

## Project structure

```text
frontend/        Streamlit user interface
src/chains/      Conversational RAG and query rewriting
src/embeddings/  Embedding model configuration
src/loaders/     Local file and web-page loaders
src/memory/      Conversation memory
src/retrieval/   Hybrid retrieval, BM25, RRF, and reranking
src/services/    Chatbot and ingestion services
src/vectorstore/ Qdrant storage operations
scripts/         Command-line ingestion script
tests/           Retrieval, service, and evaluation tests
data/            Sample knowledge-base documents
```

## Testing

Run the test suite with:

```bash
pytest
```

The `tests/` directory also includes evaluation scripts for retrieval thresholds and RAG quality.

## Roadmap

- [ ] Add streaming responses
- [ ] Add persistent per-user chat histories
- [ ] Support additional document formats and metadata filters
- [ ] Add automated CI checks

## Contributing

Contributions are welcome. Please read [CONTRIBUTING.md](CONTRIBUTING.md) before opening an issue or pull request.

## Security

Please do not report vulnerabilities in public issues. See [SECURITY.md](SECURITY.md) for the reporting process.


