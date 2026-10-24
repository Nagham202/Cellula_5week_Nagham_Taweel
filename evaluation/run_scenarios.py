import json
import tempfile
from pathlib import Path

from src.assistant import CodingAssistant
from src.code_runner import extract_python_code, run_code
from src.feedback import forget_solution
from src.memory import ConversationMemory

EVAL_DIR = Path(__file__).resolve().parent
RESULTS_PATH = EVAL_DIR / "results_scenarios.json"

CSV_SOLUTION = """Here is the correct solution:
```python
import pandas as pd


def csv_to_json(csv_path: str, json_path: str) -> None:
    # read the table, then write one JSON object per row
    table = pd.read_csv(csv_path)
    table.to_json(json_path, orient="records", indent=2)
```"""

SCENARIOS = [
    {"name": "1. Explain code (no RAG)", "turns": [
        {"message": "Explain this code:\nwords = ['a', 'bb', 'ccc']\nlengths = {w: len(w) for w in words}",
         "expect_route": "explain"},
    ]},
    {"name": "2. Generate + run", "turns": [
        {"message": "Write a function that returns the n-th Fibonacci number",
         "expect_route": "generate", "run_code": True},
    ]},
    {"name": "3. Not relevant -> teach -> ask again", "turns": [
        {"message": "Write a function that converts a CSV file to JSON using pandas",
         "expect_route": "not_relevant"},
        {"message": CSV_SOLUTION, "expect_route": "learn"},
        {"message": "Use pandas to turn a CSV file into a JSON file",
         "expect_route": "generate", "expect_in_trace": "user/"},
    ]},
    {"name": "4. Follow-up that needs memory", "turns": [
        {"message": "Write a function that counts the vowels in a string", "expect_route": "generate"},
        {"message": "now make it count only uppercase vowels", "expect_route": "generate",
         "expect_in_trace": "Follow-up rewritten as"},
        {"message": "explain the code you just wrote in one short paragraph", "expect_route": "explain"},
    ]},
]


def run_turn(assistant, turn):
    result = assistant.handle(turn["message"])
    answer = "".join(result["answer_stream"])
    checks = {"route": result["route"] == turn["expect_route"]}

    if "expect_in_trace" in turn:
        checks["trace"] = any(turn["expect_in_trace"] in step for step in result["trace"])

    if turn.get("run_code"):
        run = run_code(extract_python_code(answer))
        checks["code runs"] = run["return_code"] == 0

    return result, answer, checks


if __name__ == "__main__":
    assistant = CodingAssistant()
    report = []
    learned_ids = []

    with tempfile.TemporaryDirectory() as temp_dir:
        for scenario in SCENARIOS:
            assistant.memory = ConversationMemory(Path(temp_dir) / f"memory_{len(report)}.json")
            assistant.pending_request = None
            print("=" * 70 + f"\n{scenario['name']}")

            for turn in scenario["turns"]:
                result, answer, checks = run_turn(assistant, turn)
                ok = all(checks.values())
                print(f"  {'PASS' if ok else 'FAIL'}  {turn['message'].splitlines()[0][:60]}")
                print(f"        route={result['route']}  checks={checks}")
                for step in result["trace"]:
                    print(f"        - {step}")
                    if step.startswith("Embedded"):
                        learned_ids.append(step.split(" as ")[1].split(" ")[0])

                report.append({"scenario": scenario["name"], "message": turn["message"],
                               "route": result["route"], "trace": result["trace"],
                               "checks": checks, "passed": ok, "answer": answer})

    for doc_id in learned_ids:
        forget_solution(doc_id, assistant.vector_store)
        print(f"\n(cleanup: removed {doc_id} again)")

    passed = sum(item["passed"] for item in report)
    print(f"\nTurns passed: {passed}/{len(report)}")
    RESULTS_PATH.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved to {RESULTS_PATH.name}")
