from src.memory.conversation_memory import ConversationMemory
from src.chains.query_rewriter import rewrite_query


def main():

    memory = ConversationMemory()

    # First turn
    memory.add_user_message(
        "How many days can employees work remotely?"
    )

    memory.add_ai_message(
        "Employees can work remotely three days per week."
    )

    # Follow-up question
    question = "What about the working hours?"

    history = memory.get_history()

    print("\n==============================")
    print("QUERY REWRITER TEST")
    print("==============================")

    print("\nConversation:")
    for message in history:
        print(
            f"{message['role'].upper()}: "
            f"{message['content']}"
        )

    print("\nFollow-up question:")
    print(question)

    rewritten = rewrite_query(
        question=question,
        conversation_history=history,
    )

    print("\nRewritten question:")
    print(rewritten)


if __name__ == "__main__":
    main()