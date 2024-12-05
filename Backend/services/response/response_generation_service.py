from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def generate_general_response(user_query, message_history):
    """
    Generate a response without document context for general questions.
    """
    prompt = f"Answer the following question as a general AI response:\n{user_query}"
    full_message_history = message_history + [{"role": "user", "content": prompt}]
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=full_message_history
        )
        return response.choices[0].message.content, "Bot"
    except Exception as e:
        print(f"Error generating general response: {e}")
        return "There was an error processing your request. Please try again later.", "Bot"

def generate_response(retrieved_docs, user_query, message_history):
    """
    Generate a response using context from retrieved documents.
    """
    context = "\n\n".join([doc[1] for doc in retrieved_docs])
    sources = ", ".join([doc[0] for doc in retrieved_docs])

    prompt = (
        f"Given the following context:\n{context}\n\n"
        f"Answer the question: {user_query}\n\n"
        "Please provide a structured and well-organized response.\n"
    )
    full_message_history = message_history + [{"role": "user", "content": prompt}]

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=full_message_history
        )
        response_text = response.choices[0].message.content
        return response_text + f"\n\n---\n**Source(s):** {sources}", sources
    except Exception as e:
        print(f"Error generating response: {e}")
        return "There was an error processing your request. Please try again later.", sources
