from langsmith import traceable

from src.services.llm import get_llm


@traceable(name="Query Rewriting")
def rewrite_query(question: str, conversation_history: list) -> str:
    """
    Convert a follow-up question into a standalone question
    using the conversation history.
    """

    if not conversation_history:
        return question

    history_text = ""

    for message in conversation_history:
        role = message["role"]
        content = message["content"]

        history_text += f"{role}: {content}\n"

    prompt = f"""
You are a query rewriting assistant for a RAG chatbot.

Your job is to convert the user's latest question into a
standalone question that can be understood without conversation history.

Conversation history:
{history_text}

Latest user question:
{question}

Rules:
1. Use the conversation history only when necessary.
2. Resolve references such as "it", "they", "that", "what about".
3. Do not answer the question.
4. Return ONLY the rewritten standalone question.

Standalone question:
"""

    llm = get_llm()

    response = llm.invoke(prompt)

    return response.content.strip()