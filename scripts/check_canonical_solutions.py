import time

from src.code_runner import run_humaneval_tests
from src.knowledge_base import load_humaneval

if __name__ == "__main__":
    start = time.time()
    problems = load_humaneval()
    failed = []

    for problem in problems:
        solution = problem["prompt"] + problem["canonical_solution"]
        passed, result = run_humaneval_tests(problem, solution)
        if not passed:
            failed.append(problem["task_id"])
            print(f"FAILED {problem['task_id']}: {result['stderr'].strip()[-200:]}")

    passed_count = len(problems) - len(failed)
    print(f"\n{passed_count}/{len(problems)} canonical solutions passed  ({time.time() - start:.1f}s)")
