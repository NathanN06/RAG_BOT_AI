def precision_at_k(retrieved, relevant, k):
    if not retrieved or not relevant:
        return 0.0

    k = min(k, len(retrieved)) if k is not None else len(retrieved)

    retrieved_at_k = retrieved[:k]
    relevant_docs_in_top_k = len(set(retrieved_at_k) & relevant)

    return relevant_docs_in_top_k / k if k > 0 else 0.0


def average_precision(retrieved, relevant):
    """
    Compute Average Precision (AP) for retrieved documents.
    Considers all retrieved documents and computes precision at relevant ranks.
    """
    if not relevant:
        return 0.0

    total_precision = 0.0
    relevant_count = 0

    for i, doc_id in enumerate(retrieved):
        if doc_id in relevant:
            relevant_count += 1
            total_precision += relevant_count / (i + 1)  # Precision at rank i+1

    return total_precision / len(relevant) if relevant_count > 0 else 0.0
