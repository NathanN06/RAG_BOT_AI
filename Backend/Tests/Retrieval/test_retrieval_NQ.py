import os
import json
import warnings
import faulthandler
import pandas as pd
from beir import util
from Backend.services.embedding.embedding_service import embed_documents, embed_query
from Backend.services.Retrieval.indexing_service import index_embeddings, save_index
from Backend.services.Retrieval.retrieval_service import retrieve_documents
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from config import INDEX_FOLDER, INDEX_FILENAME
from Backend.services.embedding.embedding_service import model
from utils.chunking import chunk_text, semantic_chunk_text, recursive_chunk_text, adaptive_chunk_text, chunk_by_paragraph, chunk_by_tokens

# Enable faulthandler for better debugging
faulthandler.enable()

# Suppress specific resource warnings
warnings.filterwarnings("ignore", message="resource_tracker: There appear to be.*")

# Set the directory for saving JSON results
result_dir = "/Users/nathannguyen/Documents/RAG_BOT_1/Backend/Test_results/NQ/Automatic_metrics"
os.makedirs(result_dir, exist_ok=True)

# Choose the chunking method
CHUNKING_METHOD = "semantic"  # Set to "semantic", "recursive", etc., based on preference

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
    """
    Calculate precision, recall, and embedding similarity for the given texts.

    Parameters:
        relevant_texts (list): List of ground-truth relevant texts.
        retrieved_texts (list): List of retrieved texts.
        embeddings_model (object): Embedding model used to calculate text similarity.

    Returns:
        dict: A dictionary containing precision, recall, and embedding similarity.
    """
    print(f"Calculating metrics. Number of relevant texts: {len(relevant_texts)}, retrieved texts: {len(retrieved_texts)}")

    # Binary ground truth labels
    y_true = [1] * len(relevant_texts) + [0] * max(0, len(retrieved_texts) - len(relevant_texts))
    y_pred = [1 if text in relevant_texts else 0 for text in retrieved_texts[:len(y_true)]]
    
    # Calculate precision and recall
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    
    # Calculate embedding similarity
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

# Load NQ dataset from the provided paths
data_path = "/Users/nathannguyen/Documents/RAG_BOT_1/Backend/Data/Natural_Questions"

# Limit constants
MAX_QUERIES = 300
MAX_DOCUMENTS = 10000

# Load queries from queries.jsonl
queries_path = os.path.join(data_path, "queries.jsonl")
queries = {}
with open(queries_path, "r") as f:
    for line in f:
        query = json.loads(line)
        queries[query["_id"]] = query["text"]

# Limit to 300 queries
queries_limited = dict(list(queries.items())[:MAX_QUERIES])
print(f"Processing {len(queries_limited)} queries (limited to {MAX_QUERIES}).")

# Load corpus from corpus.jsonl
corpus_path = os.path.join(data_path, "corpus.jsonl")
corpus = {}
with open(corpus_path, "r") as f:
    for line in f:
        doc = json.loads(line)
        corpus[doc["_id"]] = {"text": doc["text"]}

# Limit to 100,000 documents
corpus_limited = dict(list(corpus.items())[:MAX_DOCUMENTS])
print(f"Processing {len(corpus_limited)} documents (limited to {MAX_DOCUMENTS}).")

# Load relevance judgments from test.tsv
test_path = os.path.join(data_path, "test.tsv")
relevance_df = pd.read_csv(test_path, sep="\t", header=None, names=["query_id", "doc_id", "relevance"])

# Filter relevance judgments for limited queries
qrels = relevance_df.groupby("query_id")["doc_id"].apply(list).to_dict()
qrels_limited = {query_id: doc_ids for query_id, doc_ids in qrels.items() if query_id in queries_limited}
print(f"Filtered relevance judgments for {len(qrels_limited)} queries.")

# Chunk and embed each corpus passage, then create the FAISS index
print("Chunking and embedding passages...")
passage_texts = []
for doc_id, doc in corpus_limited.items():
    chunks = apply_chunking_method(doc["text"], CHUNKING_METHOD)
    passage_texts.extend((f"{doc_id}_{i}", chunk) for i, chunk in enumerate(chunks))

passage_embeddings = embed_documents(passage_texts)
print("Finished embedding passages.")

print("Indexing embeddings with a Flat index for small dataset...")
index = index_embeddings(passage_embeddings, index_type='Flat')
index_path = os.path.join(INDEX_FOLDER, INDEX_FILENAME)
save_index(index, index_path)
print("Indexing complete.")

# Initialize results dictionary for retrieval and non-automated metrics
retrieval_results = {}
non_automated_metrics = {}

# Test retrieval with each query in the limited dataset
print("Running retrieval...")
for query_id, query_text in queries_limited.items():
    if len(query_text.split()) < 3:
        print(f"Skipping short query: {query_text}")
        continue

    query_embedding = embed_query(query_text)
    retrieved_docs = retrieve_documents(index, query_embedding, passage_texts, user_query=query_text)
    retrieved_doc_ids = [doc[0] for doc in retrieved_docs]

    print(f"Query ID {query_id}: Retrieved document IDs: {retrieved_doc_ids}")
    retrieval_results[query_id] = retrieved_doc_ids

    # Collect query text and retrieved document texts for non-automated metrics
    retrieved_texts = [corpus_limited[doc_id.split('_')[0]]["text"] for doc_id in retrieved_doc_ids if doc_id.split('_')[0] in corpus_limited]
    non_automated_metrics[query_id] = {
        "query_text": query_text,
        "retrieved_texts": retrieved_texts
    }

print("Retrieval complete.")

# Save non-automated metrics to JSON for review
with open(os.path.join(result_dir, "Non_automated_metric_nq.json"), "w") as f:
    json.dump(non_automated_metrics, f, indent=4)

# Save retrieval results to JSON for review
with open(os.path.join(result_dir, "retrieval_results_nq.json"), "w") as f:
    json.dump(retrieval_results, f, indent=4)

# Evaluation
print("Evaluating retrieval results...")
evaluation_results = {}
total_precision, total_recall, total_similarity = 0, 0, 0

for query_id, retrieved_doc_ids in retrieval_results.items():
    ground_truth_ids = qrels_limited.get(query_id, [])
    relevant_texts = [corpus_limited[doc_id]["text"] for doc_id in ground_truth_ids if doc_id in corpus_limited]
    retrieved_texts = [corpus_limited[doc_id.split('_')[0]]["text"] for doc_id in retrieved_doc_ids if doc_id.split('_')[0] in corpus_limited]

    eval_result = calculate_metrics(relevant_texts, retrieved_texts, embeddings_model=model)
    evaluation_results[query_id] = eval_result

    total_precision += eval_result["precision"]
    total_recall += eval_result["recall"]
    total_similarity += eval_result["embedding_similarity"]

# Compute and save averages
query_count = len(queries_limited)
average_results = {
    "average_precision": total_precision / query_count,
    "average_recall": total_recall / query_count,
    "average_similarity": total_similarity / query_count
}
print(f"Overall Averages: {average_results}")
with open(os.path.join(result_dir, "average_results.json"), "w") as f:
    json.dump(average_results, f, indent=4)

print("All results saved.")
