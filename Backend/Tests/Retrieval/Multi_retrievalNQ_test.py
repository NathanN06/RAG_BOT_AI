import os
import json
import warnings
import faulthandler
import pandas as pd
from beir import util
from services.embedding.embedding_service import embed_documents, embed_query
from services.retrieval.indexing_service import index_embeddings, save_index
from services.retrieval.retrieval_service import retrieve_documents
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from config import INDEX_FOLDER, INDEX_FILENAME
from Backend.services.embedding.embedding_service import model
from utils.chunking import chunk_text, semantic_chunk_text, recursive_chunk_text, adaptive_chunk_text, chunk_by_paragraph, chunk_by_tokens
from Tests.Retrieval.Manual_Average_precision import precision_at_k, average_precision
from Tests.Retrieval.Manual_Recall import recall_at_k
from Tests.Retrieval.Manual_Average_similarity import average_similarity

# Enable faulthandler for better debugging
faulthandler.enable()

# Suppress specific resource warnings
warnings.filterwarnings("ignore", message="resource_tracker: There appear to be.*")

# Set directories for saving JSON results
auto_result_dir = "/Users/nathannguyen/Documents/RAG_BOT_1/Backend/Test_results/NQ/Automatic_metrics"
manual_result_dir = "/Users/nathannguyen/Documents/RAG_BOT_1/Backend/Test_results/NQ/Manual_metrics"
os.makedirs(auto_result_dir, exist_ok=True)
os.makedirs(manual_result_dir, exist_ok=True)

# Choose the chunking method
CHUNKING_METHOD = "semantic"

# Define a function to apply the chosen chunking method
def apply_chunking_method(text, method=CHUNKING_METHOD):
    if method == "character":
        return chunk_text(text)
    elif method == "semantic":
        return semantic_chunk_text(text)
    elif method == "recursive":
        return recursive_chunk_text(text)
    elif method == "adaptive":
        return adaptive_chunk_text(text)
    elif method == "paragraph":
        return chunk_by_paragraph(text)
    elif method == "token":
        return chunk_by_tokens(text)
    else:
        raise ValueError(f"Unsupported chunking method: {method}")

def calculate_metrics(relevant_texts, retrieved_texts, embeddings_model):
    print(f"Calculating metrics. Number of relevant texts: {len(relevant_texts)}, retrieved texts: {len(retrieved_texts)}")

    y_true = [1] * len(relevant_texts) + [0] * max(0, len(retrieved_texts) - len(relevant_texts))
    y_pred = [1 if text in relevant_texts else 0 for text in retrieved_texts[:len(y_true)]]

    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)

    if relevant_texts and retrieved_texts:
        relevant_embeddings = model.encode(relevant_texts, normalize_embeddings=True)
        retrieved_embeddings = model.encode(retrieved_texts, normalize_embeddings=True)
        similarity = cosine_similarity(
            np.mean(relevant_embeddings, axis=0).reshape(1, -1),
            np.mean(retrieved_embeddings, axis=0).reshape(1, -1)
        )[0][0]
    else:
        similarity = 0.0

    return {
        "precision": precision,
        "recall": recall,
        "embedding_similarity": similarity
    }

# Load NQ dataset
data_path = "/Users/nathannguyen/Documents/RAG_BOT_1/Backend/Data/Natural_Questions"

MAX_QUERIES = 300
MAX_DOCUMENTS = 10000

queries_path = os.path.join(data_path, "queries.jsonl")
queries = {}
with open(queries_path, "r") as f:
    for line in f:
        query = json.loads(line)
        queries[query["_id"]] = query["text"]

queries_limited = dict(list(queries.items())[:MAX_QUERIES])
print(f"Processing {len(queries_limited)} queries (limited to {MAX_QUERIES}).")

corpus_path = os.path.join(data_path, "corpus.jsonl")
corpus = {}
with open(corpus_path, "r") as f:
    for line in f:
        doc = json.loads(line)
        corpus[doc["_id"]] = {"text": doc["text"]}

corpus_limited = dict(list(corpus.items())[:MAX_DOCUMENTS])
print(f"Processing {len(corpus_limited)} documents (limited to {MAX_DOCUMENTS}).")

test_path = os.path.join(data_path, "test.tsv")
relevance_df = pd.read_csv(test_path, sep="\t", header=None, names=["query_id", "doc_id", "relevance"])
qrels = relevance_df.groupby("query_id")["doc_id"].apply(list).to_dict()
qrels_limited = {query_id: doc_ids for query_id, doc_ids in qrels.items() if query_id in queries_limited}
print(f"Filtered relevance judgments for {len(qrels_limited)} queries.")

# Chunk and embed passages
print("Chunking and embedding passages...")
passage_texts = []
for doc_id, doc in corpus_limited.items():
    chunks = apply_chunking_method(doc["text"], CHUNKING_METHOD)
    passage_texts.extend((f"{doc_id}_{i}", chunk) for i, chunk in enumerate(chunks))

passage_embeddings = embed_documents(passage_texts)
print("Finished embedding passages.")

print("Indexing embeddings with a Flat index...")
index = index_embeddings(passage_embeddings, index_type='Flat')
index_path = os.path.join(INDEX_FOLDER, INDEX_FILENAME)
save_index(index, index_path)
print("Indexing complete.")

# Retrieval and metrics
retrieval_results = {}
evaluation_results = {}

# Initialize manual metrics tracking
manual_total_precision_at_k, manual_total_recall_at_k = 0, 0
manual_total_avg_precision, manual_total_similarity = 0, 0

print("Running retrieval...")
for query_id, query_text in queries_limited.items():
    if len(query_text.split()) < 3:
        print(f"Skipping short query: {query_text}")
        continue

    query_embedding = embed_query(query_text)
    retrieved_docs = retrieve_documents(index, query_embedding, passage_texts, user_query=query_text)
    retrieved_doc_ids = [doc[0] for doc in retrieved_docs]

    retrieved_doc_ids_base = [doc_id.split('_')[0] for doc_id in retrieved_doc_ids]
    relevant_doc_ids = set(qrels_limited.get(query_id, []))

    # Automatic metrics
    relevant_texts = [corpus_limited[doc_id]["text"] for doc_id in relevant_doc_ids if doc_id in corpus_limited]
    retrieved_texts = [corpus_limited[doc_id]["text"] for doc_id in retrieved_doc_ids_base if doc_id in corpus_limited]
    eval_result = calculate_metrics(relevant_texts, retrieved_texts, embeddings_model=model)
    evaluation_results[query_id] = eval_result

    # Manual metrics
    manual_precision_at_k = precision_at_k(retrieved_doc_ids_base, relevant_doc_ids, None)
    manual_recall_at_k = recall_at_k(retrieved_doc_ids_base, relevant_doc_ids, None)
    manual_avg_precision = average_precision(retrieved_doc_ids_base, relevant_doc_ids)

    if relevant_texts and retrieved_texts:
        retrieved_embeddings = model.encode(retrieved_texts, normalize_embeddings=True)
        relevant_embeddings = model.encode(relevant_texts, normalize_embeddings=True)
        manual_avg_similarity = average_similarity(retrieved_embeddings, relevant_embeddings)
    else:
        manual_avg_similarity = 0.0

    manual_total_precision_at_k += manual_precision_at_k
    manual_total_recall_at_k += manual_recall_at_k
    manual_total_avg_precision += manual_avg_precision
    manual_total_similarity += manual_avg_similarity

# Compute and save averages
query_count = len(queries_limited)

average_results = {
    "average_precision": sum(d["precision"] for d in evaluation_results.values()) / query_count,
    "average_recall": sum(d["recall"] for d in evaluation_results.values()) / query_count,
    "average_similarity": sum(d["embedding_similarity"] for d in evaluation_results.values()) / query_count,
}

manual_average_results = {
    "manual_precision_at_k": manual_total_precision_at_k / query_count,
    "manual_recall_at_k": manual_total_recall_at_k / query_count,
    "manual_average_precision": manual_total_avg_precision / query_count,
    "manual_average_similarity": manual_total_similarity / query_count,
}

# Save results
with open(os.path.join(auto_result_dir, "average_results.json"), "w") as f:
    json.dump(average_results, f, indent=4)

with open(os.path.join(manual_result_dir, "manual_average_results.json"), "w") as f:
    json.dump(manual_average_results, f, indent=4)

print(f"Automatic Metrics: {average_results}")
print(f"Manual Metrics: {manual_average_results}")
print("All results saved.")
