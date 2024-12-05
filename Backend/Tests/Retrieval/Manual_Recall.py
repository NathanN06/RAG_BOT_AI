def recall_at_k(retrieved, relevant, k=None):
    """
    Compute Recall@k for retrieved documents.
    If k is None, consider all retrieved documents.
    """
    if not relevant:
        return 0.0
    if k is not None:
        retrieved = retrieved[:k]
    relevant_retrieved = len(set(retrieved) & relevant)
    return relevant_retrieved / len(relevant)
