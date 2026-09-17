from pathlib import Path
import tempfile

from src.loaders.document_loader import load_document
from src.retrieval.text_splitter import split_documents
from src.vectorstore.qdrant_store import (
    add_documents,
    calculate_file_hash,
    file_hash_exists,
)


def ingest_uploaded_files(
    uploaded_files,
    client,
    embedding_model,
):
    """Load, split, embed, and store uploaded documents."""

    if not uploaded_files:
        return {
            "chunks": 0,
            "duplicates": [],
            "uploaded": [],
        }

    total_chunks = 0
    duplicates = []
    uploaded = []

    with tempfile.TemporaryDirectory() as temp_dir:

        temp_path = Path(temp_dir)

        for uploaded_file in uploaded_files:

            # 1. Save uploaded file temporarily
            file_path = temp_path / uploaded_file.name

            with open(file_path, "wb") as file:
                file.write(uploaded_file.getbuffer())

            print(f"\nProcessing: {uploaded_file.name}")

            # 2. Calculate file hash
            file_hash = calculate_file_hash(
                str(file_path)
            )

            print(f"File hash: {file_hash}")

            # 3. Check duplicate
            if file_hash_exists(client, file_hash):

                print(
                    f"Duplicate file skipped: "
                    f"{uploaded_file.name}"
                )

                duplicates.append(
                    uploaded_file.name
                )

                continue

            # 4. Load document
            documents = load_document(
                str(file_path)
            )

            # 5. Add metadata
            for document in documents:

                document.metadata["file_hash"] = file_hash

                document.metadata["original_filename"] = (
                    uploaded_file.name
                )

            # 6. Split documents
            chunks = split_documents(documents)

            # 7. Store embeddings in Qdrant
            add_documents(
                client,
                chunks,
                embedding_model,
            )

            total_chunks += len(chunks)

            uploaded.append(
                uploaded_file.name
            )

    return {
        "chunks": total_chunks,
        "duplicates": duplicates,
        "uploaded": uploaded,
    }