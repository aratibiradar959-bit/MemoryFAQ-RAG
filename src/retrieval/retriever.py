from src.vectorstore.qdrant_store import COLLECTION_NAME


def search_documents(client, embedding_model, query: str, top_k: int = 5, score_threshold: float = 0.30):
    """
    Search Qdrant and return relevant documents.

    Steps:
    1. Convert query into embedding
    2. Search Qdrant
    3. Apply similarity threshold
    4. Remove duplicate chunks from the same source
    """

    # --------------------------------------------------
    # 1. Convert the question into an embedding
    # --------------------------------------------------

    query_vector = embedding_model.embed_query(query)


    # --------------------------------------------------
    # 2. Search Qdrant
    # --------------------------------------------------

    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=top_k,
        with_payload=True,
    ).points


    # --------------------------------------------------
    # 3. Apply similarity score threshold
    # --------------------------------------------------

    filtered_results = [
        result
        for result in results
        if result.score >= score_threshold
    ]


    # --------------------------------------------------
    # 4. Remove duplicate sources
    # --------------------------------------------------

    unique_results = []
    seen_chunks = set()
    
    for result in filtered_results:
        payload = result.payload or {}
    
        chunk_id = payload.get("chunk_id")
    
        # Use chunk_id to identify duplicates.
        # If chunk_id is missing, fall back to the Qdrant point ID.
        if chunk_id is None:
            chunk_id = str(result.id)
    
        if chunk_id in seen_chunks:
            continue
    
        seen_chunks.add(chunk_id)
        unique_results.append(result)
    
    return unique_results