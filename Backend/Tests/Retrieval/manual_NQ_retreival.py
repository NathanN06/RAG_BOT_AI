import os
import json
import warnings
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer

# Importing the metric functions from the specified files
from Tests.Retrieval.Manual_Average_precision import precision_at_k, average_precision
from Tests.Retrieval.Manual_Recall import recall_at_k
from Tests.Retrieval.Manual_Average_similarity import average_similarity

from Backend.services.embedding.embedding_service import embed_documents, embed_query
from Backend.services.Retrieval.indexing_service import index_embeddings, save_index
from Backend.services.Retrieval.retrieval_service import retrieve_documents
from Backend.services.chunking.chunking_service import apply_chunking_strategy  # Import your chunking service

# Suppress warnings
warnings.filterwarnings("ignore", message="resource_tracker: There appear to be.*")

# Configuration
result_dir = "/Users/nathannguyen/Documents/RAG_BOT_1/Backend/Test_results/NQ/Manual_metrics"
os.makedirs(result_dir, exist_ok=True)

data_path = "/Users/nathannguyen/Documents/RAG_BOT_1/Backend/Data/Natural_Questions"
MAX_QUERIES = 300
MAX_DOCUMENTS = 10000
K = None  # Rank for Precision@K and Recall@K

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")  # Use the model you used in your embedding_service

# Load queries
queries_path = os.path.join(data_path, "queries.jsonl")
queries = {}
with open(queries_path, "r") as f:
    for line in f:
        query = json.loads(line)
        queries[query["_id"]] = query["text"]

queries_limited = dict(list(queries.items())[:MAX_QUERIES])
print(f"Loaded {len(queries_limited)} queries.")

# Load corpus
corpus_path = os.path.join(data_path, "corpus.jsonl")
corpus = {}
with open(corpus_path, "r") as f:
    for line in f:
        doc = json.loads(line)
        corpus[doc["_id"]] = {"text": doc["text"]}

corpus_limited = dict(list(corpus.items())[:MAX_DOCUMENTS])
print(f"Loaded {len(corpus_limited)} documents.")

# Load relevance judgments
test_path = os.path.join(data_path, "test.tsv")
relevance_df = pd.read_csv(test_path, sep="\t", header=None, names=["query_id", "doc_id", "relevance"])
qrels = relevance_df.groupby("query_id")["doc_id"].apply(list).to_dict()
qrels_limited = {query_id: doc_ids for query_id, doc_ids in qrels.items() if query_id in queries_limited}
print(f"Loaded relevance judgments for {len(qrels_limited)} queries.")

# Embed corpus and create index using adaptive chunking
print("Chunking corpus using adaptive chunking strategy and embedding...")
passage_texts = []
for doc_id, doc in corpus_limited.items():
    # Use your custom chunking strategy
    chunks = apply_chunking_strategy("adaptive", doc["text"], chunk_size=512, overlap=50)
    passage_texts.extend((f"{doc_id}_{i}", chunk) for i, chunk in enumerate(chunks))

passage_embeddings = embed_documents(passage_texts)
index = index_embeddings(passage_embeddings, index_type='Flat')
index_path = os.path.join(result_dir, "faiss_index")
save_index(index, index_path)

# Retrieval and evaluation
retrieval_results = {}
evaluation_results = {}

print("Running retrieval and evaluation with imported metrics functions...")
total_precision_at_k, total_avg_precision, total_recall_at_k, total_similarity = 0, 0, 0, 0

for query_id, query_text in queries_limited.items():
    # Instead of embedding the query here, pass the query_text directly to retrieve_documents
    query_embedding = embed_query(query_text)
    retrieved_docs = retrieve_documents(index, query_embedding, passage_texts, user_query=query_text)
    retrieved_doc_ids = [doc[0] for doc in retrieved_docs]

    # Remove chunk identifiers from retrieved document IDs
    retrieved_doc_ids_base = [doc_id.split('_')[0] for doc_id in retrieved_doc_ids]

    relevant_doc_ids = set(qrels_limited.get(query_id, []))
    if not relevant_doc_ids:
        continue

    # Log retrieved and relevant document IDs
    print(f"Query ID: {query_id}")
    print(f"Retrieved Document IDs (with chunks): {retrieved_doc_ids}")
    print(f"Retrieved Document IDs (base): {retrieved_doc_ids_base}")
    print(f"Relevant Document IDs: {relevant_doc_ids}")

    # Calculate metrics using the imported functions
    precision_at_k_val = precision_at_k(retrieved_doc_ids_base, relevant_doc_ids, K)
    recall_at_k_val = recall_at_k(retrieved_doc_ids_base, relevant_doc_ids, K)
    avg_precision_val = average_precision(retrieved_doc_ids_base, relevant_doc_ids)

    # Log calculated metrics
    print(f"Precision at K for Query ID {query_id}: {precision_at_k_val}")
    print(f"Recall at K for Query ID {query_id}: {recall_at_k_val}")
    print(f"Average Precision for Query ID {query_id}: {avg_precision_val}")

    retrieved_texts = [corpus_limited[doc_id]["text"] for doc_id in retrieved_doc_ids_base if doc_id in corpus_limited]
    relevant_texts = [corpus_limited[doc_id]["text"] for doc_id in relevant_doc_ids if doc_id in corpus_limited]

    if retrieved_texts and relevant_texts:
        retrieved_embeddings = [model.encode([text])[0] for text in retrieved_texts]
        relevant_embeddings = [model.encode([text])[0] for text in relevant_texts]
        avg_similarity_val = average_similarity(retrieved_embeddings, relevant_embeddings)
    else:
        avg_similarity_val = 0.0

    # Log similarity calculations
    print(f"Average Similarity for Query ID {query_id}: {avg_similarity_val}")

    # Store evaluation results for each query
    eval_result = {
        "precision_at_k": float(precision_at_k_val),
        "recall_at_k": float(recall_at_k_val),
        "average_precision": float(avg_precision_val),
        "average_similarity": float(avg_similarity_val)
    }
    evaluation_results[query_id] = eval_result

    # Accumulate metrics for overall calculation
    total_precision_at_k += precision_at_k_val
    total_recall_at_k += recall_at_k_val
    total_avg_precision += avg_precision_val
    total_similarity += avg_similarity_val

# Save results in the updated directory
with open(os.path.join(result_dir, "retrieval_results.json"), "w") as f:
    json.dump(retrieval_results, f, indent=4)

with open(os.path.join(result_dir, "evaluation_results.json"), "w") as f:
    json.dump(evaluation_results, f, indent=4)

# Calculate overall metrics
query_count = len(evaluation_results)
overall_metrics = {
    "mean_precision_at_k": float(total_precision_at_k / query_count),
    "mean_recall_at_k": float(total_recall_at_k / query_count),
    "mean_average_precision": float(total_avg_precision / query_count),
    "mean_average_similarity": float(total_similarity / query_count)
}

print(f"Overall Metrics with Updated Manual Code: {overall_metrics}")
with open(os.path.join(result_dir, "updated_manual_overall_metrics.json"), "w") as f:
    json.dump(overall_metrics, f, indent=4)

print("All results saved.")