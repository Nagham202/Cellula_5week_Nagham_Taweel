from src.config import TOP_K
from src.knowledge_base import load_vector_store


def retrieve(query, vector_store, k=TOP_K):
    return vector_store.similarity_search_with_score(query, k=k)


def format_context(documents):
    parts = []
    for number, document in enumerate(documents, start=1):
        label = f"[{number}] source: {document.metadata['source']}, id: {document.metadata['doc_id']}"
        parts.append(f"{label}\n{document.page_content}")
    return "\n\n---\n\n".join(parts)


if __name__ == "__main__":
    vector_store = load_vector_store()

    queries = [
        "Write a function that returns the sum of all even numbers in a list",
        "Check if a string is a palindrome",
        "Find the largest prime factor of a number",
        "Build a Flask REST API with a /users endpoint",
        "Create a CNN model in PyTorch for image classification",
    ]

    for query in queries:
        print(f"\n=== {query}")
        for document, distance in retrieve(query, vector_store):
            title = document.page_content.splitlines()[0]
            print(f"{distance:.3f}  [{document.metadata['source']}]  {title}")
