from src.memory.conversation_memory import ConversationMemory


def main():

    memory = ConversationMemory()

    # First conversation turn
    memory.add_user_message(
        "How many days can employees work remotely?"
    )

    memory.add_ai_message(
        "Employees can work remotely three days per week."
    )

    # Second conversation turn
    memory.add_user_message(
        "What about the working hours?"
    )

    memory.add_ai_message(
        "The normal working hours are 9 AM to 6 PM."
    )

    print("\n==============================")
    print("CONVERSATION HISTORY")
    print("==============================")

    for message in memory.get_history():

        print(
            f"{message['role'].upper()}: "
            f"{message['content']}"
        )


if __name__ == "__main__":
    main()