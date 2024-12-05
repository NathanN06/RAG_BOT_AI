# sparse_retrieval_service.py
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def sparse_retrieve(query, documents, top_n=10):
    """
    Retrieve top n documents using TF-IDF vectorization and cosine similarity.

    Args:
        query (str): User's query.
        documents (list): List of document texts.
        top_n (int): Number of top documents to retrieve.

    Returns:
        list: Top n documents that best match the query.
    """
    vectorizer = TfidfVectorizer(stop_words='english')
    all_texts = [doc[1] for doc in documents]  # Extract text from documents
    tfidf_matrix = vectorizer.fit_transform(all_texts)
    query_vector = vectorizer.transform([query])
    cosine_similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()
    top_indices = np.argsort(cosine_similarities)[-top_n:][::-1]

    return [documents[i] for i in top_indices]
