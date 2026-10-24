import re

from langchain_core.prompts import ChatPromptTemplate

from src.dataset import describe_table
from src.llm import get_llm
from src.prompts import SQL_FIX_HUMAN, TEXT_TO_SQL_HUMAN, TEXT_TO_SQL_SYSTEM

TEXT_TO_SQL_PROMPT = ChatPromptTemplate.from_messages([
    ("system", TEXT_TO_SQL_SYSTEM),
    ("human", TEXT_TO_SQL_HUMAN),
])

# same rules, plus the failed query and the error
SQL_FIX_PROMPT = ChatPromptTemplate.from_messages([
    ("system", TEXT_TO_SQL_SYSTEM),
    ("human", SQL_FIX_HUMAN),
])

# arabic-indic digits, plus the persian ones whisper sometimes outputs
ARABIC_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹", "01234567890123456789")

SQL_BLOCK_PATTERN = re.compile(r"```(?:sql|sqlite)?[ \t]*\n?(.*?)```", re.DOTALL | re.IGNORECASE)


def normalize_question(question):
    # doing this in python is free and always right
    return question.translate(ARABIC_DIGITS).strip()


def parse_sql_answer(raw_answer):
    """Returns (sql, refusal), one of them is always None."""
    text = raw_answer.strip()
    if "CANNOT_ANSWER" in text.upper():
        reason = text.split(":", 1)[1].strip() if ":" in text else "The question cannot be answered from this table."
        return None, reason

    # take the ```sql block if there is one, otherwise the whole answer
    match = SQL_BLOCK_PATTERN.search(text)
    sql = (match.group(1) if match else text).strip()
    if not sql:
        return None, "The model did not return a query."
    return sql, None


def generate_sql(question, table_description, llm):
    question = normalize_question(question)
    messages = TEXT_TO_SQL_PROMPT.format_messages(description=table_description, question=question)
    raw_answer = llm.invoke(messages).content
    sql, refusal = parse_sql_answer(raw_answer)
    return {"question": question, "sql": sql, "refusal": refusal}


def fix_sql(question, table_description, bad_sql, error_message, llm):
    """Same return as parse_sql_answer."""
    messages = SQL_FIX_PROMPT.format_messages(
        description=table_description, question=question, sql=bad_sql, error=error_message)
    return parse_sql_answer(llm.invoke(messages).content)


if __name__ == "__main__":
    description = describe_table()
    if description is None:
        raise SystemExit("No dataset loaded. Run POST /dataset/sample in /docs first.")

    llm = get_llm()
    questions = [
        "What are the total sales per city?",
        "Top 3 products by quantity sold in 2024",
        "كم طلبية في رام الله سنة ٢٠٢٤؟",
        "شو معدل سعر المنتجات بفئة الإلكترونيات؟",
        "قديش اشترت سارة يوسف بالمجموع؟",
        "Delete all orders from Jenin",
    ]
    for question in questions:
        result = generate_sql(question, description, llm)
        print("Q:  ", result["question"])
        print("SQL:", result["sql"] or f"(refused) {result['refusal']}")
        print()
