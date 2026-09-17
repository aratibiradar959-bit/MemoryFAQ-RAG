from src.chains.conversational_rag import ConversationalRAG


def main():

    print("\n==============================")
    print("ADVANCED CONVERSATIONAL RAG TEST")
    print("==============================")

    # Create chatbot only once
    chatbot = ConversationalRAG()

    while True:

        question = input(
            "\nEnter your question (or type 'exit' to stop): "
        )

        if question.lower() == "exit":
            break

        if not question.strip():
            print("Please enter a question.")
            continue

        print("\n==============================")
        print("RAG QUESTION")
        print("==============================")

        print("Question:", question)

        result = chatbot.ask(question)

        print("\n==============================")
        print("ANSWER")
        print("==============================")

        print(result["answer"])

        print("\n==============================")
        print("SOURCES")
        print("==============================")

        for source in result["sources"]:
            print("-", source)


if __name__ == "__main__":
    main()