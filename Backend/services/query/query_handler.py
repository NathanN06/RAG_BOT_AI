# services/query/query_handler.py

from services.embedding.embedding_service import embed_query
from services.query.query_expansion_service import expand_query
from services.persistence.index_persistence_service import load_index
from services.retrieval.document_loader import load_documents
from services.retrieval.retrieval_service import retrieve_documents
from services.response.response_generation_service import generate_general_response, generate_response
from services.retrieval.sparse_retrieval_service import sparse_retrieve
from services.query.self_query_service import generate_self_queries
from config import DATA_FOLDER, INDEX_FILENAME

def handle_query(user_query, message_history, index_filename=INDEX_FILENAME, data_folder=DATA_FOLDER):
    # Detect multiple queries in user input (Multi Query)
    if "?" in user_query or ";" in user_query:
        from services.query.multi_query_handler import handle_multi_query
        return handle_multi_query(user_query, message_history), []

    # Determine if document context is needed based on the query
    context_keywords = ["based on documents", "with context", "refer to sources"]
    use_context = any(keyword in user_query.lower() for keyword in context_keywords)

    # If no document context is required, generate a general AI response
    if not use_context:
        return generate_general_response(user_query, message_history)

    # Load the FAISS index and documents
    index = load_index(index_filename)
    documents = load_documents(data_folder)

    # Enrich the Query with Self Queries if Needed (Short Queries)
    if len(user_query.split()) < 5:  # Assume short queries need enrichment
        self_queries = generate_self_queries(user_query)
        all_retrieved_docs = []

        # Retrieve documents for each self-query
        for self_query in self_queries:
            self_query_embedding = embed_query(self_query)
            retrieved_docs = retrieve_documents(index, self_query_embedding, documents)
            all_retrieved_docs.extend(retrieved_docs)

        # Generate a response using all enriched retrieved documents
        return generate_response(all_retrieved_docs, user_query, message_history)

    # Original logic for embedding, retrieval, and response generation follows...
    # Step 1: Embed User Query
    query_embedding = embed_query(user_query)

    # Step 2: Query Expansion for Short Queries
    expanded_query_embedding = None
    if len(user_query.split()) < 5:  # Expand only for short queries
        expanded_query = expand_query(user_query)
        expanded_query_embedding = embed_query(expanded_query)

    # Step 3: Construct Combined Embedding if Query Was Expanded
    if expanded_query_embedding is not None:
        combined_embedding = (0.7 * query_embedding) + (0.3 * expanded_query_embedding)
    else:
        combined_embedding = query_embedding

    # Step 4: Perform Sparse Retrieval
    sparse_top_docs = sparse_retrieve(user_query, documents, top_n=10)

    # Step 5: Retrieve Documents Using Combined Embedding
    retrieved_docs = retrieve_documents(index, combined_embedding, sparse_top_docs, user_query)

    # Step 6: Generate Response
    if retrieved_docs:
        return generate_response(retrieved_docs, user_query, message_history)
    else:
        return "No relevant documents found", []
