# services/query/self_query_service.py

from openai import OpenAI
import os

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def generate_self_queries(user_query, num_self_queries=2):
    """
    Generate additional self queries based on the user's initial question.

    Args:
        user_query (str): The user's initial query.
        num_self_queries (int): Number of additional self-queries to generate.

    Returns:
        list: A list of additional self-queries.
    """
    prompt = f"Given the user question: '{user_query}', suggest {num_self_queries} additional sub-questions that will help provide a richer and more detailed answer."

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )
    self_queries = response.choices[0].message.content.split("\n")
    return [query.strip() for query in self_queries if query.strip()]
