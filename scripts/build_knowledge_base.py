import time

from src.config import CHROMA_DIR, HELD_OUT_PATH
from src.feedback import load_saved_records, make_feedback_document
from src.knowledge_base import add_documents, build_knowledge_base, open_vector_store

if __name__ == "__main__":
    start = time.time()
    total, knowledge, held_out, chunks = build_knowledge_base()

    print(f"Loaded {total} HumanEval problems")
    print(f"  {knowledge} -> knowledge base ({chunks} chunks in Chroma)")
    print(f"  {held_out} -> held out for testing, saved to {HELD_OUT_PATH.name}")

    # add back what users taught it, otherwise a rebuild loses it
    records = load_saved_records()
    if records:
        documents = [make_feedback_document(record) for record in records]
        user_chunks = add_documents(open_vector_store(), documents)
        print(f"  {len(records)} user-taught solutions added back ({user_chunks} chunks)")

    print(f"Database saved in {CHROMA_DIR}  ({time.time() - start:.1f}s)")
