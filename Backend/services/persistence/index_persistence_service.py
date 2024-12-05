import os
import faiss

# Define the full path for the FAISS index file (for convenience)
from config import INDEX_FOLDER, INDEX_FILENAME

INDEX_PATH = os.path.join(INDEX_FOLDER, INDEX_FILENAME)

def save_index(index, filename=INDEX_PATH):
    """
    Saves the FAISS index to the specified file.

    Args:
        index (faiss.Index): The FAISS index to save.
        filename (str): Path to save the index file.
    """
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    faiss.write_index(index, filename)

def load_index(filename=INDEX_PATH):
    """
    Loads the FAISS index from the specified file.

    Args:
        filename (str): Path to load the index file.

    Returns:
        faiss.Index: Loaded FAISS index.
    """
    if not os.path.exists(filename):
        raise FileNotFoundError(f"The FAISS index file was not found at: {filename}")
    return faiss.read_index(filename)
