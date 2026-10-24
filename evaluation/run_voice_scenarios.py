import json
import tempfile
from pathlib import Path

from fastapi.testclient import TestClient

import main
from src.config import AUDIO_TESTS_DIR
from src.memory import ConversationMemory

EVAL_DIR = Path(__file__).resolve().parent
RESULTS_PATH = EVAL_DIR / "results_voice_scenarios.json"

SCENARIOS = [
    {"name": "1. English voice -> data analysis", "steps": [
        {"voice": "en_1.m4a", "expect_route": "data_analysis", "expect_value": "Nablus"},
    ]},
    {"name": "2. Arabic voice -> data analysis", "steps": [
        {"voice": "ar_1.m4a", "expect_route": "data_analysis", "expect_value": 26397},
    ]},
    {"name": "3. Voice question + typed follow-up (memory)", "steps": [
        {"voice": "en_1.m4a", "expect_route": "data_analysis"},
        {"chat": "now only for 2023", "expect_route": "data_analysis", "expect_in_trace": "rewritten"},
    ]},
    {"name": "4. Levantine voice with the language picker", "steps": [
        {"voice": "ar_2.m4a", "language": "ar", "expect_route": "data_analysis", "expect_value": 27000},
    ]},
    {"name": "5. Week 4 path by voice: generate + run", "steps": [
        {"voice": "en_code.aiff", "expect_route": "generate", "run_code": True},
    ]},
    {"name": "6. Week 4 explain path through the API", "steps": [
        {"chat": "Explain this code: print([x * x for x in range(3)])", "expect_route": "explain"},
    ]},
]


def send(client, step):
    if "voice" in step:
        with open(AUDIO_TESTS_DIR / step["voice"], "rb") as audio:
            response = client.post("/voice", files={"audio": (step["voice"], audio)},
                                   data={"language": step.get("language", "auto")})
    else:
        response = client.post("/chat", json={"message": step["chat"]})
    return response.json()


def check(client, step, body):
    problems = []
    if body.get("route") != step["expect_route"]:
        problems.append(f"route {body.get('route')} != {step['expect_route']}")
    if "expect_in_trace" in step and step["expect_in_trace"] not in " ".join(body.get("trace", [])).lower():
        problems.append(f"'{step['expect_in_trace']}' not in trace")
    if "expect_value" in step:
        values = [value for row in (body.get("data") or {}).get("rows", []) for value in row]
        if step["expect_value"] not in values:
            problems.append(f"{step['expect_value']!r} not in result rows")
    if step.get("run_code"):
        # actually call the generated function to make sure it works
        run = client.post("/run", json={"code": body.get("code", "") + "\nprint(is_prime(7), is_prime(8))"}).json()
        if run["return_code"] != 0:
            problems.append(f"code failed: {run['stderr'][-200:]}")
    return problems


if __name__ == "__main__":
    results = []
    with TestClient(main.app) as client, tempfile.TemporaryDirectory() as temp_dir:
        main.app.state.assistant.memory = ConversationMemory(Path(temp_dir) / "memory.json")
        client.post("/dataset/sample")

        for scenario in SCENARIOS:
            print("=" * 70)
            print(scenario["name"])
            client.delete("/memory")
            steps = []
            for step in scenario["steps"]:
                body = send(client, step)
                problems = check(client, step, body)
                heard = body.get("transcript", {}).get("text")
                print(f"  {'PASS' if not problems else 'FAIL'}  {'🎤 ' + heard if heard else step.get('chat')}")
                print(f"        route={body.get('route')}  sql={((body.get('data') or {}).get('sql') or '-')[:90]!r}")
                print(f"        answer: {(body.get('answer') or body.get('detail') or '')[:160]!r}")
                for problem in problems:
                    print(f"        problem: {problem}")
                steps.append({"step": step, "heard": heard, "route": body.get("route"), "trace": body.get("trace"),
                              "sql": (body.get("data") or {}).get("sql"), "answer": body.get("answer"),
                              "problems": problems, "passed": not problems})
            results.append({"name": scenario["name"], "passed": all(s["passed"] for s in steps), "steps": steps})

    passed = sum(r["passed"] for r in results)
    print("=" * 70)
    print(f"Scenarios passed: {passed}/{len(results)}")
    RESULTS_PATH.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved to {RESULTS_PATH.name}")
