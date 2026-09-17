def reciprocal_rank_fusion(
    ranked_lists,
    k: int = 60,
):
    """
    Combine multiple ranked document lists
    using Reciprocal Rank Fusion (RRF).

    Each document should have a unique
    'chunk_id' in its metadata.

    Args:
        ranked_lists:
            List of ranked document lists.

        k:
            RRF constant. Usually 60.

    Returns:
        Documents sorted by RRF score.
    """

    rrf_scores = {}

    for ranked_list in ranked_lists:

        for rank, document in enumerate(
            ranked_list,
            start=1,
        ):

            # ---------------------------------------------
            # Get stable chunk ID
            # ---------------------------------------------

            chunk_id = document.metadata.get(
                "chunk_id"
            )

            # Fallback if chunk_id is missing
            if chunk_id is None:

                chunk_id = document.metadata.get(
                    "source"
                )

            # ---------------------------------------------
            # Calculate RRF score
            # ---------------------------------------------

            score = 1 / (k + rank)

            # ---------------------------------------------
            # Add document
            # ---------------------------------------------

            if chunk_id not in rrf_scores:

                rrf_scores[chunk_id] = {
                    "document": document,
                    "score": 0.0,
                }

            rrf_scores[chunk_id][
                "score"
            ] += score

    # ---------------------------------------------
    # Sort by RRF score
    # ---------------------------------------------

    ranked_results = sorted(
        rrf_scores.values(),
        key=lambda item: item["score"],
        reverse=True,
    )

    return ranked_results