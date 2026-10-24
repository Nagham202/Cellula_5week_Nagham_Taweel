import re

from langchain_core.prompts import ChatPromptTemplate

from src.config import EXPLAIN_ROWS_LIMIT
from src.prompts import RESULT_EXPLAINER_HUMAN, RESULT_EXPLAINER_SYSTEM

RESULT_EXPLAINER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", RESULT_EXPLAINER_SYSTEM),
    ("human", RESULT_EXPLAINER_HUMAN),
])

ARABIC_LETTERS = re.compile(r"[؀-ۿ]")


def answer_language(question):
    # decided here, not by the llm: any arabic letter means answer in arabic
    return "Arabic" if ARABIC_LETTERS.search(question) else "English"


def result_as_text(columns, rows):
    lines = [" | ".join(columns)]
    lines.extend(" | ".join(str(value) for value in row) for row in rows[:EXPLAIN_ROWS_LIMIT])
    return "\n".join(lines)


def stream_result_explanation(question, result, llm):
    row_info = f"{result['row_count']} rows"
    if result["truncated"] or result["row_count"] > EXPLAIN_ROWS_LIMIT:
        row_info += f", cut: you only see the first {min(result['row_count'], EXPLAIN_ROWS_LIMIT)}"

    messages = RESULT_EXPLAINER_PROMPT.format_messages(
        language=answer_language(question),
        question=question,
        sql=result["sql"],
        row_info=row_info,
        table=result_as_text(result["columns"], result["rows"]),
    )
    # streamed like the week 4 answers
    for piece in llm.stream(messages):
        yield piece.content


def explain_result(question, result, llm):
    return "".join(stream_result_explanation(question, result, llm))
