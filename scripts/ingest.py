import sys
from pathlib import Path

# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


from src.loaders.document_loader import load_documents
from src.retrieval.text_splitter import split_documents
from src.embeddings.embedding_model import get_embedding_model

from src.vectorstore.qdrant_store import (
    get_qdrant_client,
    create_collection,
    add_documents,
    calculate_file_hash,
    file_hash_exists,
)


DOCUMENT_FOLDER = "data"


def main():

    print("\n==============================")
    print("STARTING RAG INGESTION")
    print("==============================\n")

    # --------------------------------------------------
    # 1. Load documents
    # --------------------------------------------------

    print("Step 1: Loading documents...")

    documents = load_documents(DOCUMENT_FOLDER)

    print(f"Documents loaded: {len(documents)}\n")

    if not documents:
        print("No documents found.")
        return

    # --------------------------------------------------
    # 2. Add file metadata
    # --------------------------------------------------

    print("Step 2: Adding file metadata...")

    for document in documents:

        source = document.metadata.get("source")

        if not source:
            print("Warning: Document source not found.")
            continue

        file_path = Path(source)

        if not file_path.exists():
            print(
                f"Warning: File not found: {file_path}"
            )
            continue

        file_hash = calculate_file_hash(
            str(file_path)
        )

        document.metadata["file_hash"] = file_hash

        document.metadata["original_filename"] = (
            file_path.name
        )

    print("File metadata added.\n")

    # --------------------------------------------------
    # 3. Split documents
    # --------------------------------------------------

    print("Step 3: Splitting documents...")

    chunks = split_documents(documents)

    print(f"Chunks created: {len(chunks)}\n")

    # --------------------------------------------------
    # 4. Load embedding model
    # --------------------------------------------------

    print("Step 4: Loading embedding model...")

    embedding_model = get_embedding_model()

    print("Embedding model loaded.\n")

    # --------------------------------------------------
    # 5. Connect to Qdrant
    # --------------------------------------------------

    print("Step 5: Connecting to Qdrant...")

    client = get_qdrant_client()

    create_collection(client)

    print("Connected to Qdrant.\n")

    # --------------------------------------------------
    # 6. Check duplicate files
    # --------------------------------------------------

    print("Step 6: Checking duplicate files...")

    unique_chunks = []
    skipped_files = set()

    for chunk in chunks:

        file_hash = chunk.metadata.get(
            "file_hash"
        )

        original_filename = chunk.metadata.get(
            "original_filename"
        )

        if not file_hash:
            print(
                f"Warning: Missing hash for "
                f"{original_filename}"
            )

            unique_chunks.append(chunk)
            continue

        if file_hash in skipped_files:
            continue

        if file_hash_exists(
            client,
            file_hash,
        ):
            print(
                f"Duplicate file skipped: "
                f"{original_filename}"
            )

            skipped_files.add(file_hash)

    # Remove chunks belonging to duplicate files
    for chunk in chunks:

        file_hash = chunk.metadata.get(
            "file_hash"
        )

        if file_hash in skipped_files:
            continue

        unique_chunks.append(chunk)

    print(
        f"Chunks ready for storage: "
        f"{len(unique_chunks)}\n"
    )

    # --------------------------------------------------
    # 7. Store embeddings
    # --------------------------------------------------

    print("Step 7: Storing embeddings in Qdrant...")

    add_documents(
        client,
        unique_chunks,
        embedding_model,
    )

    # --------------------------------------------------
    # Completed
    # --------------------------------------------------

    print("\n==============================")
    print("INGESTION COMPLETED SUCCESSFULLY")
    print("==============================\n")


if __name__ == "__main__":
    main()