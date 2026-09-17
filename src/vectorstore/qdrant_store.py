from qdrant_client import QdrantClient

from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
)

from langchain_core.documents import Document

import uuid
import hashlib


COLLECTION_NAME = "memory_faq"
VECTOR_SIZE = 384


# ============================================================
# QDRANT CLIENT
# ============================================================

def get_qdrant_client():
    """Create a local Qdrant client."""

    return QdrantClient(
        path="data/qdrant"
    )


# ============================================================
# CREATE COLLECTION
# ============================================================

def create_collection(client):
    """
    Create the Qdrant collection only if it does not already exist.

    Existing vectors are preserved.
    """

    collections = client.get_collections().collections

    collection_names = [
        collection.name
        for collection in collections
    ]

    # Collection already exists
    if COLLECTION_NAME in collection_names:

        print(
            f"Collection '{COLLECTION_NAME}' already exists."
        )

        return

    # Create collection
    client.create_collection(

        collection_name=COLLECTION_NAME,

        vectors_config=VectorParams(
            size=VECTOR_SIZE,
            distance=Distance.COSINE,
        ),
    )

    print(
        f"Collection '{COLLECTION_NAME}' created."
    )


# ============================================================
# FILE HASH
# ============================================================

def calculate_file_hash(file_path: str) -> str:
    """
    Calculate SHA-256 hash of a file.

    Used to detect duplicate local files.
    """

    sha256 = hashlib.sha256()

    with open(file_path, "rb") as file:

        while chunk := file.read(8192):

            sha256.update(chunk)

    return sha256.hexdigest()


# ============================================================
# WEB HASH
# ============================================================

def calculate_web_hash(url: str) -> str:
    """
    Calculate SHA-256 hash of a webpage URL.

    Used to identify duplicate web pages.
    """

    return hashlib.sha256(
        url.encode("utf-8")
    ).hexdigest()


# ============================================================
# FILE HASH EXISTS
# ============================================================

def file_hash_exists(
    client,
    file_hash: str,
) -> bool:
    """
    Check whether a local file already exists
    in Qdrant using its SHA-256 hash.
    """

    results = client.scroll(

        collection_name=COLLECTION_NAME,

        scroll_filter=Filter(
            must=[
                FieldCondition(

                    key="file_hash",

                    match=MatchValue(
                        value=file_hash,
                    ),
                )
            ]
        ),

        limit=1,

        with_payload=True,
    )

    points, _ = results

    return len(points) > 0


# ============================================================
# WEB HASH EXISTS
# ============================================================

def web_hash_exists(
    client,
    web_hash: str,
) -> bool:
    """
    Check whether a webpage already exists
    in Qdrant using its URL hash.
    """

    results = client.scroll(

        collection_name=COLLECTION_NAME,

        scroll_filter=Filter(
            must=[
                FieldCondition(

                    key="web_hash",

                    match=MatchValue(
                        value=web_hash,
                    ),
                )
            ]
        ),

        limit=1,

        with_payload=True,
    )

    points, _ = results

    return len(points) > 0


# ============================================================
# ADD DOCUMENTS
# ============================================================

def add_documents(
    client,
    documents,
    embedding_model,
):
    """
    Create embeddings for document chunks
    and store them in Qdrant.

    Supports both:

    1. Local documents
    2. Web documents

    Local documents use:
        file_hash

    Web documents use:
        web_hash

    Each chunk receives a stable chunk_id.
    """

    points = []

    for index, document in enumerate(documents):

        # --------------------------------------------------
        # 1. Create embedding
        # --------------------------------------------------

        vector = embedding_model.embed_query(
            document.page_content
        )

        # --------------------------------------------------
        # 2. Get metadata
        # --------------------------------------------------

        metadata = document.metadata

        file_type = metadata.get(
            "file_type"
        )

        source = metadata.get(
            "source"
        )

        url = metadata.get(
            "url"
        )

        file_hash = metadata.get(
            "file_hash"
        )

        original_filename = metadata.get(
            "original_filename"
        )

        # --------------------------------------------------
        # 3. Create identity
        # --------------------------------------------------

        web_hash = metadata.get(
            "web_hash"
        )

        # --------------------------------------------------
        # Web document
        # --------------------------------------------------

        if file_type == "web":

            # If web_hash is missing,
            # create it from the URL.

            if not web_hash and url:

                web_hash = calculate_web_hash(
                    url
                )

            # Create stable web chunk ID

            chunk_id = (
                f"{web_hash}_{index}"
                if web_hash
                else f"web_{uuid.uuid4()}"
            )

        # --------------------------------------------------
        # Local document
        # --------------------------------------------------

        else:

            chunk_id = (
                f"{file_hash}_{index}"
                if file_hash
                else f"document_{uuid.uuid4()}"
            )

        # --------------------------------------------------
        # 4. Create Qdrant point
        # --------------------------------------------------

        point = PointStruct(

            # Qdrant internal ID
            id=str(uuid.uuid4()),

            # Embedding vector
            vector=vector,

            # Metadata / payload
            payload={

                # Original text
                "text": document.page_content,

                # Common metadata
                "source": source,

                "file_type": file_type,

                # Web metadata
                "url": url,

                "web_hash": web_hash,

                # Local file metadata
                "file_hash": file_hash,

                "original_filename": (
                    original_filename
                ),

                # Stable chunk identifier
                "chunk_id": chunk_id,
            },
        )

        points.append(point)

    # ------------------------------------------------------
    # 5. Store points
    # ------------------------------------------------------

    if points:

        client.upsert(

            collection_name=COLLECTION_NAME,

            points=points,
        )

        print(
            f"Stored {len(points)} chunks in Qdrant."
        )


# ============================================================
# GET ALL CHUNKS
# ============================================================

def get_all_chunks(client):
    """
    Get all document chunks from Qdrant.

    Converts Qdrant payloads back into
    LangChain Document objects.

    Used by BM25 sparse retrieval.
    """

    documents = []

    offset = None

    while True:

        points, next_offset = client.scroll(

            collection_name=COLLECTION_NAME,

            limit=100,

            offset=offset,

            with_payload=True,

            with_vectors=False,
        )

        for point in points:

            payload = point.payload or {}

            document = Document(

                # Original chunk text
                page_content=payload.get(
                    "text",
                    "",
                ),

                # Metadata
                metadata={

                    "source": payload.get(
                        "source"
                    ),

                    "file_type": payload.get(
                        "file_type"
                    ),

                    "url": payload.get(
                        "url"
                    ),

                    "web_hash": payload.get(
                        "web_hash"
                    ),

                    "file_hash": payload.get(
                        "file_hash"
                    ),

                    "original_filename": (
                        payload.get(
                            "original_filename"
                        )
                    ),

                    "chunk_id": payload.get(
                        "chunk_id"
                    ),
                },
            )

            documents.append(
                document
            )

        if next_offset is None:

            break

        offset = next_offset

    return documents


# ============================================================
# GET ALL DOCUMENTS
# ============================================================

def get_all_documents(client):
    """
    Get all unique documents from Qdrant.

    Supports both:

    - Local files
    - Web pages

    Returns a list containing:

    Local:
        filename
        file_hash
        file_type
        chunk_count

    Web:
        filename
        url
        web_hash
        file_type
        chunk_count
    """

    documents = {}

    offset = None

    while True:

        points, next_offset = client.scroll(

            collection_name=COLLECTION_NAME,

            limit=100,

            offset=offset,

            with_payload=True,

            with_vectors=False,
        )

        for point in points:

            payload = point.payload or {}

            file_type = payload.get(
                "file_type"
            )

            # --------------------------------------------------
            # WEB DOCUMENT
            # --------------------------------------------------

            if file_type == "web":

                web_hash = payload.get(
                    "web_hash"
                )

                url = payload.get(
                    "url"
                )

                # Older web records may not have
                # web_hash.

                if not web_hash and url:

                    web_hash = calculate_web_hash(
                        url
                    )

                if not web_hash:

                    continue

                document_id = (
                    f"web_{web_hash}"
                )

                if document_id not in documents:

                    documents[document_id] = {

                        "filename": (
                            url
                            or "Web Page"
                        ),

                        "url": url,

                        "web_hash": web_hash,

                        "file_hash": None,

                        "file_type": "web",

                        "chunk_count": 0,
                    }

                documents[
                    document_id
                ]["chunk_count"] += 1

            # --------------------------------------------------
            # LOCAL DOCUMENT
            # --------------------------------------------------

            else:

                file_hash = payload.get(
                    "file_hash"
                )

                filename = payload.get(
                    "original_filename"
                )

                # Skip points without
                # file information

                if not file_hash:

                    continue

                document_id = (
                    f"file_{file_hash}"
                )

                if document_id not in documents:

                    documents[document_id] = {

                        "filename": (
                            filename
                            or "Unknown"
                        ),

                        "url": None,

                        "web_hash": None,

                        "file_hash": file_hash,

                        "file_type": (
                            file_type
                            or "Unknown"
                        ),

                        "chunk_count": 0,
                    }

                documents[
                    document_id
                ]["chunk_count"] += 1

        if next_offset is None:

            break

        offset = next_offset

    return list(
        documents.values()
    )


# ============================================================
# DELETE LOCAL DOCUMENT
# ============================================================

def delete_document(
    client,
    file_hash: str,
):
    """
    Delete all chunks belonging to a local document
    using its file hash.
    """

    client.delete(

        collection_name=COLLECTION_NAME,

        points_selector=Filter(

            must=[

                FieldCondition(

                    key="file_hash",

                    match=MatchValue(

                        value=file_hash,
                    ),
                )
            ]
        ),
    )

    print(
        f"Deleted document with hash: {file_hash}"
    )


# ============================================================
# DELETE WEB DOCUMENT
# ============================================================

def delete_web_document(
    client,
    web_hash: str,
):
    """
    Delete all chunks belonging to a webpage
    using its web hash.
    """

    client.delete(

        collection_name=COLLECTION_NAME,

        points_selector=Filter(

            must=[

                FieldCondition(

                    key="web_hash",

                    match=MatchValue(

                        value=web_hash,
                    ),
                )
            ]
        ),
    )

    print(
        f"Deleted web document with hash: {web_hash}"
    )