import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from src.config import MAX_OUTPUT_CHARS, RUN_TIMEOUT_SECONDS

# only python blocks, things like ```bash are skipped
CODE_BLOCK_PATTERN = re.compile(r"```(?:python|py)?[ \t]*\n(.*?)```", re.DOTALL)


def extract_python_code(markdown_text):
    blocks = CODE_BLOCK_PATTERN.findall(markdown_text)
    return "\n\n".join(block.strip() for block in blocks)


def to_text(output):
    if output is None:
        return ""
    # after a timeout python sometimes gives bytes back
    if isinstance(output, bytes):
        output = output.decode("utf-8", errors="replace")
    if len(output) > MAX_OUTPUT_CHARS:
        output = output[:MAX_OUTPUT_CHARS] + "\n... (output cut)"
    return output


def run_code(code, timeout=RUN_TIMEOUT_SECONDS):
    with tempfile.TemporaryDirectory(prefix="code_runner_") as temp_dir:
        script_path = Path(temp_dir) / "main.py"
        script_path.write_text(code, encoding="utf-8")

        # clean env so the code can't see my api key
        safe_env = {"PATH": "/usr/bin:/bin", "HOME": temp_dir}

        command = [
            sys.executable,
            # isolated mode, ignores my own packages and settings
            "-I",
            "-X", "utf8",
            str(script_path),
        ]

        start = time.time()
        try:
            completed = subprocess.run(
                command,
                cwd=temp_dir,
                env=safe_env,
                # input() fails right away instead of hanging
                stdin=subprocess.DEVNULL,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            return {
                "stdout": to_text(completed.stdout),
                "stderr": to_text(completed.stderr),
                "return_code": completed.returncode,
                "timed_out": False,
                "seconds": round(time.time() - start, 2),
            }
        except subprocess.TimeoutExpired as error:
            return {
                "stdout": to_text(error.stdout),
                "stderr": to_text(error.stderr) + f"\nStopped: the code ran longer than {timeout} seconds.",
                "return_code": None,
                "timed_out": True,
                "seconds": round(time.time() - start, 2),
            }


def run_humaneval_tests(problem, solution_code):
    program = f"{solution_code}\n\n{problem['test']}\n\ncheck({problem['entry_point']})\n"
    result = run_code(program)
    passed = result["return_code"] == 0
    return passed, result


if __name__ == "__main__":
    tests = {
        "1. normal output": 'print("Hello from the code runner!")\nprint(sum(range(10)))',
        "2. an error (goes to stderr)": "print('before')\nprint(1 / 0)",
        "3. infinite loop (should time out)": "while True:\n    pass",
        "4. can it see my API key?": 'import os\nprint("GROQ_API_KEY =", os.environ.get("GROQ_API_KEY"))',
        "5. where does it run?": "import os\nprint(os.getcwd())\nopen('note.txt', 'w').write('hi')",
    }
    for name, code in tests.items():
        result = run_code(code, timeout=3)
        print(f"=== {name}")
        print(f"return_code={result['return_code']}  timed_out={result['timed_out']}  ({result['seconds']}s)")
        print("stdout:", result["stdout"].strip() or "-")
        print("stderr:", result["stderr"].strip().splitlines()[-1] if result["stderr"].strip() else "-")
        print()

    answer = "Here you go:\n```python\nprint('extracted!')\n```\nRun it with:\n```bash\npython main.py\n```"
    print("=== 6. extract code from a Markdown answer")
    print(repr(extract_python_code(answer)))
