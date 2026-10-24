import json
import re
from datetime import datetime

from src.code_runner import extract_python_code
from src.config import MEMORY_CODE_LIMIT, MEMORY_PATH, MEMORY_TEXT_LIMIT, MEMORY_TURNS_IN_PROMPT

LANGUAGES = {
    "python": "Python", "javascript": "JavaScript", "typescript": "TypeScript", "java": "Java",
    "c++": "C++", "c#": "C#", "golang": "Go", "rust": "Rust", "sql": "SQL",
}
FRAMEWORKS = {
    "flask": "Flask", "fastapi": "FastAPI", "django": "Django", "pytorch": "PyTorch",
    "tensorflow": "TensorFlow", "keras": "Keras", "scikit-learn": "scikit-learn", "sklearn": "scikit-learn",
    "pandas": "pandas", "numpy": "NumPy", "streamlit": "Streamlit", "langchain": "LangChain", "react": "React",
}


# whole words only, so java doesn't match inside javascript
def mentions(word, text):
    return re.search(rf"(?<!\w){re.escape(word)}(?!\w)", text) is not None


def shorten(text, limit):
    return text if len(text) <= limit else text[:limit] + " ...(cut)"


class ConversationMemory:
    def __init__(self, path=MEMORY_PATH):
        self.path = path
        if path.exists():
            self.data = json.loads(path.read_text(encoding="utf-8"))
        else:
            self.data = self.empty()

    @staticmethod
    def empty():
        return {"turns": [], "generated_code": [], "preferences": {"language": None, "framework": None}}

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.data, indent=2, ensure_ascii=False), encoding="utf-8")

    def clear(self):
        self.data = self.empty()
        self.save()

    def update_preferences(self, message):
        text = message.lower()
        for word, name in LANGUAGES.items():
            if mentions(word, text):
                self.data["preferences"]["language"] = name
        for word, name in FRAMEWORKS.items():
            if mentions(word, text):
                self.data["preferences"]["framework"] = name

    def add_turn(self, user_message, route, answer):
        self.update_preferences(user_message)
        self.data["turns"].append({
            "time": datetime.now().isoformat(timespec="seconds"),
            "user": user_message,
            "route": route,
            "assistant": answer,
        })
        # kept separately so "explain the code you just wrote" works
        code = extract_python_code(answer)
        if route == "generate" and code:
            self.data["generated_code"].append({"request": user_message, "code": code})
        self.save()

    def last_code(self):
        if not self.data["generated_code"]:
            return None
        return self.data["generated_code"][-1]["code"]

    def as_prompt_text(self, include_code=True):
        if not self.data["turns"]:
            return "Conversation memory: no previous conversation."

        preferences = self.data["preferences"]
        lines = [
            "Conversation memory (use it for follow-ups; ignore it if the new message is unrelated):",
            f"User preferences: language = {preferences['language'] or 'not set'}, "
            f"framework = {preferences['framework'] or 'not set'}",
            "Recent conversation:",
        ]
        # only the last few turns, to keep the prompt small
        for turn in self.data["turns"][-MEMORY_TURNS_IN_PROMPT:]:
            lines.append(f"- User: {shorten(turn['user'], MEMORY_TEXT_LIMIT)}")
            lines.append(f"  Assistant ({turn['route']}): {shorten(turn['assistant'], MEMORY_TEXT_LIMIT)}")

        code = self.last_code()
        if include_code and code:
            lines.append("Last generated code:")
            lines.append("```python\n" + shorten(code, MEMORY_CODE_LIMIT) + "\n```")
        return "\n".join(lines)
