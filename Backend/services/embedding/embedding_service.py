# embedding_service.py
import numpy as np
from Models.embedding_model import model

def embed_documents(documents):
    """
    Embeds a list of document texts.
    
    Args:
        documents (list): A list of tuples where each tuple contains a document identifier and text.
    
    Returns:
        np.ndarray: An array of embeddings corresponding to the document texts.
    """
    texts = [text for _, text in documents]
    embeddings = model.encode(texts)
    return embeddings

def embed_query(query, additional_context=None, average_embeddings=False):
    """
    Embeds a single user query with optional context expansion and averaging.
    
    Args:
        query (str): The user's query string.
        additional_context (list, optional): List of additional context strings to enrich the query.
        average_embeddings (bool, optional): Whether to average multiple embeddings of query variations.
    
    Returns:
        np.ndarray: An array of embeddings corresponding to the query.
    """
    # Basic embedding of the query
    queries = [query]

    # Add additional context for query expansion if provided
    if additional_context:
        queries += [f"{query} {context}" for context in additional_context]

    # Embed all variations of the query
    query_embeddings = model.encode(queries)

    # Return averaged embedding if requested
    if average_embeddings and len(query_embeddings) > 1:
        return np.mean(query_embeddings, axis=0)
    
    # Otherwise, return the embedding of the original query
    return query_embeddings[0] if not average_embeddings else query_embeddings
