from sentence_transformers import SentenceTransformer
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Initialize the model only once to improve performance
model = SentenceTransformer('all-MiniLM-L6-v2')
