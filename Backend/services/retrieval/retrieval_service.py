import numpy as np
from services.query.query_expansion_service import expand_query
from services.embedding.embedding_service import embed_query
from services.response.reranking_service import rerank_with_model
from sklearn.metrics.pairwise import cosine_similarity

def retrieve_documents(index, query_embedding, documents, user_query=None, k=10, nprobe=20):
    """
    Retrieve the top k relevant documents based on embedding similarity,
    then refine relevance through reranking using the CrossEncoder model.

    Args:
        index (faiss.Index): The FAISS index object.
        query_embedding (np.ndarray): The embedding vector of the query.
        documents (list): List of document tuples containing metadata and text.
        user_query (str): The original user query text.
        k (int): The number of top documents to retrieve.
        nprobe (int): The number of clusters to probe for initial retrieval.

    Returns:
        list: List of retrieved and reranked document tuples.
    """
    index.nprobe = nprobe  # Increase nprobe to broaden initial retrieval

    # Ensure query_embedding has the shape (1, d)
    if query_embedding.ndim == 1:
        query_embedding = query_embedding.reshape(1, -1)

    # Expand the query if it is short, using synonyms and related terms
    expanded_query_embedding = None
    if user_query and len(user_query.split()) < 5:  # Expand only for short queries
        expanded_query = expand_query(user_query)
        expanded_query_embedding = embed_query(expanded_query)
        if expanded_query_embedding.ndim == 1:
            expanded_query_embedding = expanded_query_embedding.reshape(1, -1)

    # Calculate an initial similarity with a few top documents to set adaptive weights
    _, initial_indices = index.search(query_embedding, k=3)
    initial_docs = [documents[i] for i in initial_indices[0] if 0 <= i < len(documents) and len(documents[i]) > 2]

    # Calculate initial similarity if there are documents to compare
    initial_sim = 0
    if initial_docs:
        initial_sim = np.mean([
            cosine_similarity(query_embedding, [doc[1]])[0][0]
            for doc in initial_docs if len(doc) > 1
        ])

    # Set weights based on the initial similarity threshold
    initial_sim_threshold = 0.8
    weight_original = 0.9 if initial_sim >= initial_sim_threshold else 0.7
    weight_expanded = 1 - weight_original

    # Construct the combined embedding adaptively based on the presence of expanded embedding
    if expanded_query_embedding is not None:
        combined_embedding = (weight_original * query_embedding +
                              weight_expanded * expanded_query_embedding)
    else:
        combined_embedding = query_embedding

    # Step 1: Perform the initial retrieval from FAISS with larger k for more coverage
    _, indices = index.search(combined_embedding, k)
    indices = indices.flatten()  # Flatten indices for easier access

    # Step 2: Retrieve documents corresponding to the indices
    retrieved_docs = []
    for idx in indices:
        if 0 <= idx < len(documents):
            retrieved_docs.append(documents[idx])
        else:
            print(f"Warning: Index {idx} is out of range for documents list of length {len(documents)}.")

    if not retrieved_docs:
        print("No valid document indices found after initial retrieval.")
        return []

    # Step 3: Apply reranking using the CrossEncoder model, focusing on top candidates
    if user_query and retrieved_docs:
        reranked_docs = rerank_with_model(retrieved_docs[:5], user_query)  # Limit reranking to top 5
        return reranked_docs
    else:
        return retrieved_docs
