import argparse
import json
import time
from pathlib import Path

from src.code_runner import extract_python_code, run_humaneval_tests
from src.config import HELD_OUT_PATH
from src.generator import stream_generation
from src.knowledge_base import load_vector_store
from src.llm import get_llm
from src.relevance_judge import RELEVANT, judge_relevance
from src.retriever import retrieve

EVAL_DIR = Path(__file__).resolve().parent
RESULTS_PATH = EVAL_DIR / "results_pass_at_1.json"


def make_request(problem):
    return ("Complete this Python function. Keep exactly the same function name and parameters.\n"
            f"```python\n{problem['prompt']}\n```")


def solve(problem, llm, vector_store):
    request = make_request(problem)
    documents = [document for document, distance in retrieve(request, vector_store)]
    retrieved_ids = [document.metadata["doc_id"] for document in documents]

    verdict, raw_answer = judge_relevance(request, documents, llm)
    if verdict != RELEVANT:
        return verdict, retrieved_ids, ""

    answer = "".join(stream_generation(request, documents, llm))
    return verdict, retrieved_ids, extract_python_code(answer)


def failure_reason(verdict, code, passed, run):
    if verdict != RELEVANT:
        return "judge said Not Relevant"
    if not code:
        return "no code block in the answer"
    if passed:
        return ""
    if run["timed_out"]:
        return "timeout"
    last_line = run["stderr"].strip().splitlines()[-1] if run["stderr"].strip() else "unknown error"
    return last_line[:120]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="only the first N held-out problems")
    args = parser.parse_args()

    problems = json.loads(HELD_OUT_PATH.read_text(encoding="utf-8"))[:args.limit]
    llm = get_llm()
    vector_store = load_vector_store()
    results = []

    for number, problem in enumerate(problems, start=1):
        start = time.time()
        verdict, retrieved_ids, code = solve(problem, llm, vector_store)

        passed, run = False, None
        if code:
            passed, run = run_humaneval_tests(problem, code)

        leaked = problem["task_id"] in retrieved_ids
        reason = failure_reason(verdict, code, passed, run)
        mark = "PASS" if passed else "FAIL"
        print(f"{number:2d}. {mark} {problem['task_id']:14s} judge={verdict:12s} "
              f"{time.time() - start:5.1f}s  {reason}")

        results.append({"task_id": problem["task_id"], "entry_point": problem["entry_point"],
                        "verdict": verdict, "retrieved": retrieved_ids, "leaked": leaked,
                        "passed": passed, "reason": reason, "code": code})

    total = len(results)
    passed_count = sum(r["passed"] for r in results)
    relevant_count = sum(r["verdict"] == RELEVANT for r in results)
    print(f"\npass@1: {passed_count}/{total} = {passed_count / total:.0%}")
    print(f"judge said Relevant for {relevant_count}/{total} held-out problems")
    print(f"leaked (held-out problem found in Chroma): {sum(r['leaked'] for r in results)}")

    RESULTS_PATH.write_text(json.dumps({"pass_at_1": passed_count / total, "results": results},
                                       indent=2), encoding="utf-8")
    print(f"Saved to {RESULTS_PATH.name}")
