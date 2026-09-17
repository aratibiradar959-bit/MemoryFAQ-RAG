
from langchain_core.documents import Document

from src.retrieval.reranker import CrossEncoderReranker


documents = [
    Document(
        page_content="Employees can work remotely three days per week.",
        metadata={
            "source": "remote_work_policy.txt"
        },
    ),

    Document(
        page_content="The company provides health insurance to employees.",
        metadata={
            "source": "benefits.txt"
        },
    ),

    Document(
        page_content="The working hours are 9 AM to 6 PM.",
        metadata={
            "source": "company_info.txt"
        },
    ),

    Document(
        page_content="The yearly learning allowance is 15,000 rupees.",
        metadata={
            "source": "new_policy.txt"
        },
    ),
]


query = "How many days can employees work remotely?"


print("=" * 70)
print("                 CROSS-ENCODER RERANKING")
print("=" * 70)

print("\nQuery:")
print(query)


reranker = CrossEncoderReranker()


results = reranker.rerank(
    query=query,
    documents=documents,
    top_k=3,
)


print("\n" + "-" * 70)

for rank, result in enumerate(
    results,
    start=1,
):

    document = result["document"]
    score = result["score"]

    print(f"Rank: {rank}")
    print(
        f"Score: {score:.4f}"
    )
    print(
        f"Source: {document.metadata.get('source')}"
    )
    print(
        f"Text: {document.page_content}"
    )

    print("\n" + "-" * 70)


print("\nCross-Encoder test completed.")