from utils.chunking import (
    chunk_text, semantic_chunk_text, recursive_chunk_text, 
    adaptive_chunk_text, chunk_by_paragraph, chunk_by_tokens
)

def apply_chunking_strategy(chunk_strategy, text, chunk_size=512, overlap=0):
    """
    Apply the specified chunking strategy to a given text.

    Args:
        chunk_strategy (str): Chunking strategy to use ('character', 'semantic', 'recursive', 'adaptive', 'paragraph', 'token').
        text (str): The text to be chunked.
        chunk_size (int): Maximum size of each chunk.
        overlap (int): Number of overlapping characters between chunks.

    Returns:
        list: List of chunked text.
    """
    if chunk_strategy == 'semantic':
        return semantic_chunk_text(text, max_length=chunk_size, overlap=overlap)
    elif chunk_strategy == 'recursive':
        return recursive_chunk_text(text, max_length=chunk_size, overlap=overlap)
    elif chunk_strategy == 'adaptive':
        return adaptive_chunk_text(text, default_max_length=chunk_size, overlap=overlap)
    elif chunk_strategy == 'paragraph':
        return chunk_by_paragraph(text, max_length=chunk_size, overlap=overlap)
    elif chunk_strategy == 'token':
        return chunk_by_tokens(text, max_tokens=chunk_size, overlap=overlap)
    else:
        return chunk_text(text, max_length=chunk_size, overlap=overlap)
