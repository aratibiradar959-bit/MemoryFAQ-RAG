from src.loaders.web_loader import load_web_page
from src.retrieval.text_splitter import split_documents
from src.vectorstore.qdrant_store import (
    calculate_web_hash,
    web_hash_exists,
    add_documents,
)


def ingest_web_page(
    url: str,
    client,
    embedding_model,
):
    """
    Load a webpage, split it into chunks,
    check for duplicates, and store it in Qdrant.
    """

    # ----------------------------------------------
    # Clean URL
    # ----------------------------------------------

    url = url.strip()

    if not url:
        raise ValueError("URL cannot be empty.")

    # ----------------------------------------------
    # Calculate web hash
    # ----------------------------------------------

    web_hash = calculate_web_hash(url)

    # ----------------------------------------------
    # Check duplicate
    # ----------------------------------------------

    if web_hash_exists(client, web_hash):
        return {
            "status": "duplicate",
            "url": url,
            "chunks": 0,
        }

    # ----------------------------------------------
    # Load webpage
    # ----------------------------------------------

    documents = load_web_page(url)

    if not documents:
        raise ValueError(
            "No content could be loaded from the webpage."
        )

    # ----------------------------------------------
    # Split webpage into chunks
    # ----------------------------------------------

    chunks = split_documents(documents)

    if not chunks:
        raise ValueError(
            "No chunks were created from the webpage."
        )

    # ----------------------------------------------
    # Add web metadata
    # ----------------------------------------------

    for chunk in chunks:

        chunk.metadata["source"] = url
        chunk.metadata["url"] = url
        chunk.metadata["file_type"] = "web"
        chunk.metadata["web_hash"] = web_hash

    # ----------------------------------------------
    # Store in Qdrant
    # ----------------------------------------------

    add_documents(
        client=client,
        documents=chunks,
        embedding_model=embedding_model,
    )

    return {
        "status": "success",
        "url": url,
        "chunks": len(chunks),
    }