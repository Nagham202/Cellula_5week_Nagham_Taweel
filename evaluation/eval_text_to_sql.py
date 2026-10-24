import json
import time
from itertools import permutations
from pathlib import Path

from src.config import SAMPLE_DATASET_PATH
from src.data_analyst import analyze
from src.dataset import load_dataset
from src.llm import get_llm
from src.sql_safety import run_select

EVAL_DIR = Path(__file__).resolve().parent
TESTSET_PATH = EVAL_DIR / "text_to_sql_testset.json"
RESULTS_PATH = EVAL_DIR / "results_text_to_sql.json"


def normalize(value):
    if isinstance(value, (int, float)):
        return round(float(value), 2)
    return value


def same_result(gold, actual):
    """Compares results, not sql text. Extra columns and row order don't matter."""
    if len(gold["rows"]) != len(actual["rows"]):
        return False
    gold_rows = sorted((tuple(normalize(v) for v in row) for row in gold["rows"]), key=str)
    column_count = len(gold["columns"])
    for picked in permutations(range(len(actual["columns"])), column_count):
        actual_rows = sorted((tuple(normalize(row[i]) for i in picked) for row in actual["rows"]), key=str)
        if actual_rows == gold_rows:
            return True
    return False


if __name__ == "__main__":
    load_dataset(SAMPLE_DATASET_PATH.read_bytes(), SAMPLE_DATASET_PATH.name)
    testset = json.loads(TESTSET_PATH.read_text(encoding="utf-8"))
    llm = get_llm()
    results = []

    for number, case in enumerate(testset, start=1):
        gold = run_select(case["gold_sql"])
        start = time.time()
        # only the sql is tested, skipping the explanation saves a call
        answer = analyze(case["question"], llm, explain=False)
        seconds = time.time() - start

        correct = answer["error"] is None and same_result(gold, answer)
        print(f"{number:2d}. {'OK  ' if correct else 'MISS'} [{case['language']}] {seconds:4.1f}s  {case['question']}")
        if not correct:
            print(f"      SQL:      {(answer['sql'] or answer['error']).replace(chr(10), ' ')}")
            print(f"      got:      {answer['rows'][:3]}")
            print(f"      expected: {gold['rows'][:3]}")

        results.append({**case, "generated_sql": answer["sql"], "error": answer["error"],
                        "fixed_by_retry": any("fixed SQL" in step for step in answer["trace"]),
                        "correct": correct, "seconds": round(seconds, 2)})

    print()
    for language in ("en", "ar"):
        subset = [r for r in results if r["language"] == language]
        print(f"{language}: {sum(r['correct'] for r in subset)}/{len(subset)}")
    correct_count = sum(r["correct"] for r in results)
    accuracy = correct_count / len(results)
    print(f"Execution accuracy: {correct_count}/{len(results)} = {accuracy:.0%}")
    print(f"Fixed by the one retry: {sum(r['fixed_by_retry'] for r in results)}")

    RESULTS_PATH.write_text(json.dumps({"accuracy": accuracy, "results": results},
                                       indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved to {RESULTS_PATH.name}")
