from langchain_core.documents import Document

from src.retrieval.bm25_retriever import (
    BM25Retriever,
)


# ---------------------------------------------------------
# Sample documents
# ---------------------------------------------------------

documents = [

    Document(
        page_content=(
            "Employees can work remotely "
            "three days per week."
        ),
        metadata={
            "source": "remote_work_policy.txt"
        },
    ),

    Document(
        page_content=(
            "The working hours are "
            "9 AM to 6 PM."
        ),
        metadata={
            "source": "company_info.txt"
        },
    ),

    Document(
        page_content=(
            "The company provides "
            "health insurance to employees."
        ),
        metadata={
            "source": "benefits.txt"
        },
    ),

    Document(
        page_content=(
            "The yearly learning allowance "
            "is 15,000 rupees."
        ),
        metadata={
            "source": "new_policy.txt"
        },
    ),
]


# ---------------------------------------------------------
# Create BM25 retriever
# ---------------------------------------------------------

retriever = BM25Retriever(
    documents
)


# ---------------------------------------------------------
# Query
# ---------------------------------------------------------

query = (
    "How many days can employees "
    "work remotely?"
)


# ---------------------------------------------------------
# Search
# ---------------------------------------------------------

results = retriever.search(
    query=query,
    top_k=3,
)


# ---------------------------------------------------------
# Display results
# ---------------------------------------------------------

print("\n")
print("=" * 60)
print("                 BM25 SEARCH")
print("=" * 60)

print(
    f"\nQuery: {query}"
)


for rank, (document, score) in enumerate(
    results,
    start=1,
):

    print("\n" + "-" * 60)

    print(
        f"Rank: {rank}"
    )

    print(
        f"Score: {score:.4f}"
    )

    print(
        f"Source: "
        f"{document.metadata.get('source')}"
    )

    print(
        f"Text: "
        f"{document.page_content}"
    )