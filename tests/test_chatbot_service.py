from src.services.chatbot_service import ChatbotService


def main():

    chatbot = ChatbotService()

    print("\n==============================")
    print("CHATBOT SERVICE TEST")
    print("==============================")

    question1 = "How many days can employees work remotely?"

    result1 = chatbot.ask(question1)

    print("\nUSER:")
    print(question1)

    print("\nAI:")
    print(result1["answer"])

    print("\nSOURCES:")

    for source in result1["sources"]:
        print("-", source)

    question2 = "What about the working hours?"

    result2 = chatbot.ask(question2)

    print("\nUSER:")
    print(question2)

    print("\nAI:")
    print(result2["answer"])

    print("\nSTANDALONE QUESTION:")
    print(result2["standalone_question"])

    print("\n==============================")
    print("MEMORY")
    print("==============================")

    for message in chatbot.get_history():

        print(
            f"{message['role'].upper()}: "
            f"{message['content']}"
        )


if __name__ == "__main__":
    main()