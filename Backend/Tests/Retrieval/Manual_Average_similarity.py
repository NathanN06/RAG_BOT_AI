import numpy as np

def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:  # Avoid division by zero
        return 0.0
    return dot_product / (norm1 * norm2)

def average_similarity(retrieved_embeddings, relevant_embeddings):
    """
    Compute average cosine similarity between the mean embeddings
    of retrieved and relevant documents.
    """
    # Ensure non-empty arrays
    if retrieved_embeddings.size == 0 or relevant_embeddings.size == 0:
        return 0.0

    # Compute mean embeddings
    retrieved_mean_embedding = np.mean(retrieved_embeddings, axis=0)
    relevant_mean_embedding = np.mean(relevant_embeddings, axis=0)

    # Normalize to unit vectors
    retrieved_mean_embedding /= np.linalg.norm(retrieved_mean_embedding) or 1
    relevant_mean_embedding /= np.linalg.norm(relevant_mean_embedding) or 1

    # Cosine similarity
    return cosine_similarity(retrieved_mean_embedding, relevant_mean_embedding)
