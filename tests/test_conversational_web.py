import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.chains.conversational_rag import ConversationalRAG


def main():

    print("\n" + "=" * 60)
    print("CONVERSATIONAL WEB + LOCAL RAG TEST")
    print("=" * 60)

    rag = ConversationalRAG()

    # --------------------------------------------------
    # TEST 1: Local document
    # --------------------------------------------------

    print("\n" + "=" * 60)
    print("TEST 1: LOCAL DOCUMENT")
    print("=" * 60)

    result = rag.ask(
        "How many days can employees work remotely?"
    )

    print("\nQuestion:")
    print(result["question"])

    print("\nAnswer:")
    print(result["answer"])

    print("\nSources:")
    for source in result["sources"]:
        print("-", source)

    # --------------------------------------------------
    # TEST 2: Web document
    # --------------------------------------------------

    print("\n" + "=" * 60)
    print("TEST 2: WEB DOCUMENT")
    print("=" * 60)

    result = rag.ask(
        "What is this domain used for?"
    )

    print("\nQuestion:")
    print(result["question"])

    print("\nAnswer:")
    print(result["answer"])

    print("\nSources:")
    for source in result["sources"]:
        print("-", source)

    # --------------------------------------------------
    # TEST 3: Unsupported question
    # --------------------------------------------------

    print("\n" + "=" * 60)
    print("TEST 3: UNSUPPORTED QUESTION")
    print("=" * 60)

    result = rag.ask(
        "What is the capital of France?"
    )

    print("\nQuestion:")
    print(result["question"])

    print("\nAnswer:")
    print(result["answer"])

    print("\nSources:")
    for source in result["sources"]:
        print("-", source)

    print("\n" + "=" * 60)
    print("TEST COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    main()
    