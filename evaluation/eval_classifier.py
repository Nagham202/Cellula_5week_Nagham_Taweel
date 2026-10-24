import json
import time
from pathlib import Path

from src.classifier import classify_intent
from src.config import SAMPLE_DATASET_PATH
from src.dataset import load_dataset
from src.llm import get_llm

EVAL_DIR = Path(__file__).resolve().parent
TESTSET_PATH = EVAL_DIR / "classifier_testset.json"
RESULTS_PATH = EVAL_DIR / "results_classifier.json"

if __name__ == "__main__":
    # the classifier sees the table's columns, so always use the same table
    load_dataset(SAMPLE_DATASET_PATH.read_bytes(), SAMPLE_DATASET_PATH.name)
    testset = json.loads(TESTSET_PATH.read_text(encoding="utf-8"))
    llm = get_llm()
    results = []

    for number, case in enumerate(testset, start=1):
        start = time.time()
        predicted, raw_answer = classify_intent(case["message"], llm)
        seconds = time.time() - start

        correct = predicted == case["expected"]
        mark = "OK  " if correct else "MISS"
        first_line = case["message"].splitlines()[0][:60]
        print(f"{number:2d}. {mark} expected={case['expected']:16s} got={predicted:16s} {seconds:.2f}s  {first_line}")

        results.append({**case, "predicted": predicted, "raw_answer": raw_answer,
                        "correct": correct, "seconds": round(seconds, 2)})

    correct_count = sum(r["correct"] for r in results)
    accuracy = correct_count / len(results)
    average_seconds = sum(r["seconds"] for r in results) / len(results)
    for label in sorted({r["expected"] for r in results}):
        subset = [r for r in results if r["expected"] == label]
        print(f"  {label}: {sum(r['correct'] for r in subset)}/{len(subset)}")
    print(f"\nAccuracy: {correct_count}/{len(results)} = {accuracy:.0%}   (average {average_seconds:.2f}s per message)")

    RESULTS_PATH.write_text(json.dumps({"accuracy": accuracy, "results": results},
                                       indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved to {RESULTS_PATH.name}")
