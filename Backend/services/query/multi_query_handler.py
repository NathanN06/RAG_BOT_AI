# services/query/multi_query_handler.py

from services.persistence.index_persistence_service import load_index
from services.retrieval.document_loader import load_documents
from services.retrieval.retrieval_service import retrieve_documents
from services.embedding.embedding_service import embed_query
from services.response.response_generation_service import generate_response
from services.retrieval.sparse_retrieval_service import sparse_retrieve
from config import DATA_FOLDER, INDEX_FILENAME

def handle_multi_query(user_queries, message_history):
    """
    Handle multiple related user queries and generate a combined response.

    Args:
        user_queries (str): The user's multiple queries in one string.
        message_history (list): History of conversation messages.

    Returns:
        str: Combined response for all user queries.
    """
    # Step 1: Split user queries based on question marks or semicolons
    queries = [q.strip() for q in user_queries.split("?") if q.strip()]
    if len(queries) == 1:
        queries = [q.strip() for q in user_queries.split(";") if q.strip()]

    responses = []
    
    # Load the FAISS index and documents once, reuse for each query
    index = load_index(INDEX_FILENAME)
    documents = load_documents(DATA_FOLDER)

    # Step 2: Handle each query independently
    for query in queries:
        # Step 2.1: Embed the individual query
        query_embedding = embed_query(query)

        # Step 2.2: Perform sparse retrieval
        sparse_top_docs = sparse_retrieve(query, documents, top_n=10)

        # Step 2.3: Retrieve documents using the embedding
        retrieved_docs = retrieve_documents(index, query_embedding, sparse_top_docs, query)

        # Step 2.4: Generate response for the specific query
        if retrieved_docs:
            response, _ = generate_response(retrieved_docs, query, message_history)
        else:
            response = "No relevant documents found for this query."

        responses.append(response)

    # Step 3: Combine the individual responses
    combined_response = "\n\n".join(responses)
    return combined_response
