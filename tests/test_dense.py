from src.embeddings.embedding_model import get_embedding_model
from src.vectorstore.qdrant_store import get_qdrant_client
from src.retrieval.retriever import search_documents


def main():

    print("=" * 70)
    print("                 DENSE SEARCH DEBUG")
    print("=" * 70)

    print("\nLoading embedding model...")
    embedding_model = get_embedding_model()

    print("\nConnecting to Qdrant...")
    client = get_qdrant_client()

    query = "What are the normal working hours?"

    print("\nQuery:")
    print(query)

    results = search_documents(
        client=client,
        embedding_model=embedding_model,
        query=query,
        top_k=10,
        score_threshold=0.0,
    )

    print("\n" + "=" * 70)
    print("DENSE SEARCH RESULTS")
    print("=" * 70)

    if not results:
        print("\nNo dense search results found.")

    for rank, result in enumerate(results, start=1):

        payload = result.payload or {}

        print("\n" + "-" * 70)
        print(f"Rank: {rank}")
        print(f"Dense Score: {result.score:.6f}")
        print(f"Source: {payload.get('original_filename')}")
        print(f"Chunk ID: {payload.get('chunk_id')}")
        print(f"Text: {payload.get('text', '')[:500]}")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()