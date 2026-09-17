from src.chains.conversational_rag import ConversationalRAG


class ChatbotService:
    """High-level service for the multi-turn RAG chatbot."""

    def __init__(self):
        self.chatbot = ConversationalRAG()

    def ask(self, question: str):
        """Ask a question and return the chatbot response."""

        if not question or not question.strip():
            return {
                "answer": "Please enter a question.",
                "sources": [],
                "standalone_question": "",
            }

        result = self.chatbot.ask(question.strip())

        return {
            "answer": result["answer"],
            "sources": result["sources"],
            "standalone_question": result["standalone_question"],
        }

    def get_history(self):
        """Return conversation history."""

        return self.chatbot.memory.get_history()

    def clear_memory(self):
        """Clear the current conversation."""

        self.chatbot.memory.clear()