import os
import fitz  # PyMuPDF
import json
from services.chunking.chunking_service import apply_chunking_strategy
from config import DATA_FOLDER

def load_documents(data_folder, chunk_strategy='character', chunk_size=512, overlap=0, save_file=None):
    """
    Load and chunk documents from PDFs in the specified data folder using various chunking strategies.

    Args:
        data_folder (str): Path to the folder containing PDF files.
        chunk_strategy (str): Chunking strategy to use ('character', 'semantic', 'recursive', 'adaptive', 'paragraph', 'token').
        chunk_size (int): Maximum size of each chunk.
        overlap (int): Number of overlapping characters between chunks (applies to certain chunking strategies).
        save_file (str): File path to save the preprocessed chunks.

    Returns:
        list: List of chunked document texts.
    """
    # Set default save path if not provided
    if save_file is None:
        save_dir = "/Users/nathannguyen/Documents/RAG_BOT_1/Backend/Data/Preprocessed_Data"
        os.makedirs(save_dir, exist_ok=True)
        save_file = os.path.join(save_dir, "preprocessed_documents.json")

    documents = []

    if os.path.exists(save_file):
        print(f"Loading preprocessed documents from {save_file}")
        with open(save_file, "r") as f:
            documents = json.load(f)
        return documents  # Return preprocessed documents directly

    for filename in os.listdir(data_folder):
        if filename.endswith('.pdf'):
            pdf_path = os.path.join(data_folder, filename)
            pdf_document = fitz.open(pdf_path)  # Open the PDF document with PyMuPDF
            text = ""  # Hold extracted text from PDF
            for page in pdf_document:
                text += page.get_text()  # Extract text from each page

            # Only chunk if there's text
            if text.strip():
                chunks = apply_chunking_strategy(chunk_strategy, text, chunk_size, overlap)
                for chunk in chunks:
                    documents.append((filename, chunk))

    with open(save_file, "w") as f:
        json.dump(documents, f)

    print(f"Preprocessed documents saved to {save_file}")
    return documents
