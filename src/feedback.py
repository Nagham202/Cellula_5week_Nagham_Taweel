import hashlib
import json
from datetime import datetime

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate

from src.config import USER_SOLUTIONS_PATH
from src.knowledge_base import add_documents
from src.prompts import FEEDBACK_HUMAN, FEEDBACK_SYSTEM

FEEDBACK_PROMPT = ChatPromptTemplate.from_messages([
    ("system", FEEDBACK_SYSTEM),
    ("human", FEEDBACK_HUMAN),
])


def is_solution(pending_request, message, llm):
    messages = FEEDBACK_PROMPT.format_messages(pending_request=pending_request, message=message)
    raw_answer = llm.invoke(messages).content
    return "solution" in raw_answer.lower(), raw_answer


def make_doc_id(request, solution):
    fingerprint = hashlib.sha1((request + solution).encode("utf-8")).hexdigest()[:10]
    return f"user/{fingerprint}"


def make_feedback_document(record):
    text = f"# User-taught solution for: {record['request']}\n\n{record['solution']}"
    metadata = {
        "doc_id": record["doc_id"],
        "source": "user_feedback",
        "request": record["request"],
        "added_at": record["added_at"],
    }
    return Document(page_content=text, metadata=metadata)


def save_solution(request, solution, vector_store):
    record = {
        "doc_id": make_doc_id(request, solution),
        "request": request,
        "solution": solution,
        "added_at": datetime.now().isoformat(timespec="seconds"),
    }

    document = make_feedback_document(record)
    chunk_count = add_documents(vector_store, [document])

    # don't write the same solution to the file twice
    already_saved = {saved["doc_id"] for saved in load_saved_records()}
    if record["doc_id"] not in already_saved:
        USER_SOLUTIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(USER_SOLUTIONS_PATH, "a", encoding="utf-8") as file:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")

    return record["doc_id"], chunk_count


def forget_solution(doc_id, vector_store):
    stored = vector_store.get(where={"doc_id": doc_id})
    if stored["ids"]:
        vector_store.delete(ids=stored["ids"])

    kept = [record for record in load_saved_records() if record["doc_id"] != doc_id]
    lines = [json.dumps(record, ensure_ascii=False) + "\n" for record in kept]
    USER_SOLUTIONS_PATH.write_text("".join(lines), encoding="utf-8")
    return len(stored["ids"])


def load_saved_records():
    if not USER_SOLUTIONS_PATH.exists():
        return []
    lines = USER_SOLUTIONS_PATH.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]
