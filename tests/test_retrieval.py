from src.embeddings.embedding_model import get_embedding_model
from src.vectorstore.qdrant_store import get_qdrant_client
from src.retrieval.retriever import search_documents


def main():

    print("Loading embedding model...")
    embedding_model = get_embedding_model()

    print("Connecting to Qdrant...")
    client = get_qdrant_client()

    print("\n==============================")
    print("RAG RETRIEVAL TEST")
    print("==============================")

    while True:

        query = input(
            "\nEnter your question "
            "(or type 'exit' to stop): "
        )

        if query.lower() == "exit":
            break

        results = search_documents(
            client=client,
            embedding_model=embedding_model,
            query=query,
            top_k=5,
        )

        print("\nRetrieved Documents:")
        print("==============================")

        for index, result in enumerate(results, start=1):

            print(f"\n--- Result {index} ---")

            print("Score:", result.score)

            print("Text:")
            print(result.payload.get("text"))

            print("Source:")
            print(result.payload.get("source"))


if __name__ == "__main__":
    main()