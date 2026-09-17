from src.services.llm import get_llm


def main():

    print("Loading OpenRouter LLM...")

    llm = get_llm()

    response = llm.invoke(
        "Say hello and confirm that the OpenRouter connection is working."
    )

    print("\nLLM Response:")
    print(response.content)


if __name__ == "__main__":
    main()