from src.embeddings.embedding_model import get_embedding_model


def main():
    print("Loading embedding model...")

    embeddings = get_embedding_model()

    text = "Employees can work remotely three days per week."

    vector = embeddings.embed_query(text)

    print("Embedding created successfully!")
    print("Vector dimensions:", len(vector))
    print("First 10 values:", vector[:10])


if __name__ == "__main__":
    main()