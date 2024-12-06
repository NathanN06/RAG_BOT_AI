import os
import fitz  # PyMuPDF
import json
from services.chunking.chunking_service import apply_chunking_strategy
from config import DATA_FOLDER

def load_documents(data_folder, chunk_strategy='character', chunk_size=512, overlap=0, save_file=None, force_reprocess=False):
    """
    Load and chunk documents from PDFs in the specified data folder using various chunking strategies.

    Args:
        data_folder (str): Path to the folder containing PDF files.
        chunk_strategy (str): Chunking strategy to use ('character', 'semantic', 'recursive', 'adaptive', 'paragraph', 'token').
        chunk_size (int): Maximum size of each chunk.
        overlap (int): Number of overlapping characters between chunks (applies to certain chunking strategies).
        save_file (str): File path to save the preprocessed chunks.
        force_reprocess (bool): Whether to force reprocessing all documents.

    Returns:
        list: List of chunked document texts.
    """
    # Set default save path if not provided
    if save_file is None:
        save_dir = os.path.join(DATA_FOLDER, "Preprocessed_Data")
        os.makedirs(save_dir, exist_ok=True)
        save_file = os.path.join(save_dir, "preprocessed_documents.json")

    documents = []

    # Load preprocessed data if available and force_reprocess is False
    if os.path.exists(save_file) and not force_reprocess:
        print(f"Loading preprocessed documents from {save_file}")
        with open(save_file, "r") as f:
            documents = json.load(f)
        return documents  # Return preprocessed documents directly

    # Track already processed files to avoid duplicates, normalize to lowercase
    processed_files = {doc[0].lower() for doc in documents}

    # Process new files that haven't been processed yet
    new_documents = []
    for filename in os.listdir(data_folder):
        filename_lower = filename.lower()  # Normalize filename to lowercase for consistent processing

        if filename_lower.endswith('.pdf') and filename_lower not in processed_files:
            pdf_path = os.path.join(data_folder, filename)
            try:
                pdf_document = fitz.open(pdf_path)  # Open the PDF document with PyMuPDF
                text = ""  # Hold extracted text from PDF

                # Extract text from each page
                for page in pdf_document:
                    text += page.get_text()

                # Only chunk if there's text
                if text.strip():
                    chunks = apply_chunking_strategy(chunk_strategy, text, chunk_size, overlap)
                    for chunk in chunks:
                        print(f"Chunk from '{filename}': {chunk[:200]}...")  # Log the first 200 characters for verification
                        new_documents.append((filename_lower, chunk))

                print(f"Successfully processed: {filename}")

            except Exception as e:
                print(f"Failed to process file '{filename}': {str(e)}")

    # Only save if there are new documents processed
    if new_documents:
        documents.extend(new_documents)
        with open(save_file, "w") as f:
            json.dump(documents, f)
        print(f"Preprocessed documents saved to {save_file}")

    print(f"Total number of documents processed: {len(documents)}")
    return documents
