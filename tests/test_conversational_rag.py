from src.chains.conversational_rag import ConversationalRAG


def main():

    chatbot = ConversationalRAG()

    print("\n")
    print("==============================")
    print("MULTI-TURN RAG CHATBOT")
    print("==============================")

    # --------------------------------
    # Turn 1
    # --------------------------------

    question1 = "How many days can employees work remotely?"

    result1 = chatbot.ask(question1)

    print("\n==============================")
    print("TURN 1")
    print("==============================")

    print("User:", question1)
    print("AI:", result1["answer"])

    # --------------------------------
    # Turn 2
    # --------------------------------

    question2 = "What about the working hours?"

    result2 = chatbot.ask(question2)

    print("\n==============================")
    print("TURN 2")
    print("==============================")

    print("User:", question2)
    print("AI:", result2["answer"])

    # --------------------------------
    # Turn 3
    # --------------------------------

    question3 = "How many paid leaves do they get?"

    result3 = chatbot.ask(question3)

    print("\n==============================")
    print("TURN 3")
    print("==============================")

    print("User:", question3)
    print("AI:", result3["answer"])

    # --------------------------------
    # Conversation history
    # --------------------------------

    print("\n==============================")
    print("MEMORY")
    print("==============================")

    for message in chatbot.memory.get_history():

        print(
            f"{message['role'].upper()}: "
            f"{message['content']}"
        )


if __name__ == "__main__":
    main()