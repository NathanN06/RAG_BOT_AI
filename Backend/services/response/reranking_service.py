from sentence_transformers import CrossEncoder

# Assuming a CrossEncoder model instance (this should be loaded once, ideally)
reranker_model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

def rerank_with_model(retrieved_docs, user_query):
    """
    Reranks the initially retrieved documents using a CrossEncoder model.

    Args:
        retrieved_docs (list): List of retrieved document texts to be reranked.
        user_query (str): The user's query.

    Returns:
        list: Reranked documents in the order of their relevance.
    """
    # Create input pairs for reranker
    input_pairs = [(user_query, doc[1]) for doc in retrieved_docs]  # Assuming `doc[1]` is the document text

    # Use CrossEncoder to calculate relevance scores for each input pair
    scores = reranker_model.predict(input_pairs)

    # Combine scores with documents
    scored_docs = list(zip(retrieved_docs, scores))

    # Sort documents by their scores in descending order
    scored_docs = sorted(scored_docs, key=lambda x: x[1], reverse=True)

    # Return documents sorted by relevance score (without the scores themselves)
    return [doc for doc, _ in scored_docs]
