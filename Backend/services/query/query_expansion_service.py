import re
from nltk.corpus import wordnet
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS  # Import stop words

def expand_query(query):
    """
    Expands the query by adding synonyms or related terms to the main keywords.
    """
    keywords = extract_keywords(query)
    if len(keywords) < 3:
        expanded_terms = set(keywords)
        for keyword in keywords:
            for syn in wordnet.synsets(keyword):
                for lemma in syn.lemmas():
                    if lemma.name().lower() != keyword:
                        expanded_terms.add(lemma.name().lower())
        return " ".join(expanded_terms)
    return query

def extract_keywords(query):
    """
    Extract keywords from the query by removing stop words and non-alphanumeric characters.
    """
    words = re.findall(r'\b\w+\b', query.lower())
    keywords = [word for word in words if word not in ENGLISH_STOP_WORDS]
    return set(keywords)
