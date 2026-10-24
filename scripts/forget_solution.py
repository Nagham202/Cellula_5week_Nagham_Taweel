import sys

from src.feedback import forget_solution
from src.knowledge_base import load_vector_store

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python -m scripts.forget_solution <doc_id>")
        sys.exit(1)

    doc_id = sys.argv[1]
    removed = forget_solution(doc_id, load_vector_store())
    print(f"Removed {removed} chunk(s) of {doc_id} from Chroma and from user_solutions.jsonl")
