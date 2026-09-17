from langchain_core.documents import Document

from src.retrieval.rrf import (
    reciprocal_rank_fusion,
)


# ---------------------------------------------------------
# Documents
# ---------------------------------------------------------

remote_work = Document(
    page_content=(
        "Employees can work remotely "
        "three days per week."
    ),
    metadata={
        "source": "remote_work_policy.txt"
    },
)


company_info = Document(
    page_content=(
        "The working hours are "
        "9 AM to 6 PM."
    ),
    metadata={
        "source": "company_info.txt"
    },
)


benefits = Document(
    page_content=(
        "The company provides "
        "health insurance to employees."
    ),
    metadata={
        "source": "benefits.txt"
    },
)


learning = Document(
    page_content=(
        "The yearly learning allowance "
        "is 15,000 rupees."
    ),
    metadata={
        "source": "new_policy.txt"
    },
)


# ---------------------------------------------------------
# Dense Retrieval Ranking
# ---------------------------------------------------------

dense_results = [
    remote_work,
    company_info,
    benefits,
    learning,
]


# ---------------------------------------------------------
# BM25 Ranking
# ---------------------------------------------------------

sparse_results = [
    remote_work,
    benefits,
    company_info,
    learning,
]


# ---------------------------------------------------------
# RRF Fusion
# ---------------------------------------------------------

results = reciprocal_rank_fusion(
    ranked_lists=[
        dense_results,
        sparse_results,
    ]
)


# ---------------------------------------------------------
# Display Results
# ---------------------------------------------------------

print("\n")

print("=" * 70)

print(
    "                 RRF FUSION RESULTS"
)

print("=" * 70)


for rank, result in enumerate(
    results,
    start=1,
):

    document = result[
        "document"
    ]

    score = result[
        "score"
    ]

    source = document.metadata.get(
        "source"
    )

    print("\n" + "-" * 70)

    print(
        f"Rank: {rank}"
    )

    print(
        f"Source: {source}"
    )

    print(
        f"RRF Score: {score:.6f}"
    )

    print(
        f"Text: {document.page_content}"
    )


print("\n")

print("=" * 70)

print(
    "RRF fusion test completed."
)

print("=" * 70)

print("\n")