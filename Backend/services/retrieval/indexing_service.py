import faiss
import numpy as np
from services.embedding.embedding_service import embed_documents
from services.retrieval.document_loader import load_documents
from services.persistence.index_persistence_service import save_index, load_index
from config import DATA_FOLDER, INDEX_PATH

def index_embeddings(embeddings, index_type='IVFPQ'):
    """
    Indexes embeddings using FAISS, falling back to Flat index for smaller datasets.

    Args:
        embeddings (np.ndarray): Array of embeddings to index.
        index_type (str): Type of FAISS index to use ('IVFPQ', 'IVFFlat', etc.).

    Returns:
        faiss.Index: The FAISS index with the embeddings added.
    """
    dimension = embeddings.shape[1]
    num_data_points = embeddings.shape[0]

    # Determine the appropriate index based on the size of the dataset
    if num_data_points < 100:
        print(f"Dataset is small ({num_data_points} data points). Using IndexFlatL2.")
        index = faiss.IndexFlatL2(dimension)
    elif index_type == 'IVFPQ':
        nlist = min(10, num_data_points // 2)  # Set nlist based on data size, typically a lower value
        quantizer = faiss.IndexFlatL2(dimension)  # Quantizer for clustering
        index = faiss.IndexIVFPQ(quantizer, dimension, nlist, 8, 8)
        index.train(embeddings)
    elif index_type == 'IVFFlat':
        quantizer = faiss.IndexFlatL2(dimension)
        nlist = min(10, num_data_points // 2)
        index = faiss.IndexIVFFlat(quantizer, dimension, nlist, faiss.METRIC_L2)
        index.train(embeddings)
    else:
        index = faiss.IndexFlatL2(dimension)

    # Add embeddings to the index
    index.add(embeddings)
    return index

def create_and_save_index(data_folder=DATA_FOLDER, index_filename=INDEX_PATH, chunk_strategy='token', chunk_size=512, overlap=0):
    """
    Loads documents, creates embeddings, indexes them, and saves the index.

    Args:
        data_folder (str): Path to the folder containing documents.
        index_filename (str): Path to save the FAISS index.
        chunk_strategy (str): The chunking strategy to use.
        chunk_size (int): The size limit for each chunk.
        overlap (int): The overlap size for document chunking.
    """
    documents = load_documents(data_folder, chunk_strategy=chunk_strategy, chunk_size=chunk_size, overlap=overlap)
    embeddings = embed_documents(documents)
    index = index_embeddings(embeddings)
    save_index(index, index_filename)
