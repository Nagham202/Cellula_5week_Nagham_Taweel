import json
from pathlib import Path

from src.knowledge_base import load_vector_store
from src.llm import get_llm
from src.relevance_judge import judge_relevance
from src.retriever import retrieve

EVAL_DIR = Path(__file__).resolve().parent
TESTSET_PATH = EVAL_DIR / "relevance_testset.json"
RESULTS_PATH = EVAL_DIR / "results_relevance.json"

if __name__ == "__main__":
    testset = json.loads(TESTSET_PATH.read_text(encoding="utf-8"))
    llm = get_llm()
    vector_store = load_vector_store()
    results = []

    for number, case in enumerate(testset, start=1):
        retrieved = retrieve(case["request"], vector_store)
        documents = [document for document, distance in retrieved]
        closest_distance = retrieved[0][1]

        verdict, raw_answer = judge_relevance(case["request"], documents, llm)
        correct = verdict == case["expected"]
        mark = "OK  " if correct else "MISS"
        print(f"{number:2d}. {mark} expected={case['expected']:12s} got={verdict:12s} "
              f"closest={closest_distance:.3f}  {case['request'][:55]}")

        results.append({**case, "verdict": verdict, "raw_answer": raw_answer, "correct": correct,
                        "retrieved": [doc.metadata["doc_id"] for doc in documents],
                        "closest_distance": round(closest_distance, 3)})

    correct_count = sum(r["correct"] for r in results)
    accuracy = correct_count / len(results)
    print(f"\nAccuracy: {correct_count}/{len(results)} = {accuracy:.0%}")

    RESULTS_PATH.write_text(json.dumps({"accuracy": accuracy, "results": results}, indent=2), encoding="utf-8")
    print(f"Saved to {RESULTS_PATH.name}")
