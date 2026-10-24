import sqlite3

from src.config import CHART_MAX_POINTS
from src.dataset import describe_table
from src.llm import get_llm
from src.result_explainer import answer_language, explain_result
from src.sql_generator import fix_sql, generate_sql
from src.sql_safety import UnsafeSQLError, run_select


# same keys as a normal result so the api never has to check which one it got
def failed(question, sql, error, trace):
    return {"question": question, "sql": sql, "columns": [], "rows": [], "row_count": 0,
            "truncated": False, "chart": None, "explanation": None, "error": error, "trace": trace}


def suggest_chart(columns, rows):
    """Bar chart for label + number results, line chart if the labels are dates."""
    if len(columns) != 2 or not 2 <= len(rows) <= CHART_MAX_POINTS:
        return None
    labels_are_text = all(isinstance(row[0], str) for row in rows)
    values_are_numbers = all(isinstance(row[1], (int, float)) for row in rows)
    if not (labels_are_text and values_are_numbers):
        return None
    # months and years read better as a line
    is_time = any(word in columns[0].lower() for word in ("date", "month", "year", "day"))
    return {"type": "line" if is_time else "bar", "x": columns[0], "y": columns[1]}


def run_with_one_fix(question, description, sql, llm, trace):
    try:
        return run_select(sql)
    except sqlite3.Error as error:
        trace.append(f"SQLite error: {error} -> asking the LLM to fix the query (once)")
        # only real sql errors get a second try, blocked queries never do
        fixed_sql, refusal = fix_sql(question, description, sql, str(error), llm)
        if fixed_sql is None:
            raise
        trace.append(f"LLM fixed SQL: {fixed_sql}")
        # one retry only, no loops burning tokens
        return run_select(fixed_sql)


def analyze(question, llm, explain=True):
    """question -> sql -> safety check -> sqlite -> rows, chart and explanation"""
    trace = []
    description = describe_table()
    if description is None:
        return failed(question, None, "No dataset loaded yet. Upload one first.", trace)

    generated = generate_sql(question, description, llm)
    question = generated["question"]
    if generated["refusal"]:
        trace.append(f"LLM refused: {generated['refusal']}")
        return failed(question, None, generated["refusal"], trace)

    sql = generated["sql"]
    trace.append(f"LLM generated SQL: {sql}")
    try:
        result = run_with_one_fix(question, description, sql, llm, trace)
    except UnsafeSQLError as error:
        trace.append(f"Blocked by the safety check: {error}")
        return failed(question, sql, str(error), trace)
    except sqlite3.Error as error:
        trace.append(f"Failed: {error}")
        return failed(question, sql, f"The query failed: {error}", trace)

    trace.append(f"Safety check passed + ran read-only: {result['row_count']} rows in {result['seconds']}s")

    chart = suggest_chart(result["columns"], result["rows"])
    trace.append(f"Chart: {chart['type']} ({chart['x']} vs {chart['y']})" if chart else "Chart: none (result not chartable)")

    explanation = None
    # the evals pass explain=False to save a call
    if explain:
        explanation = explain_result(question, result, llm)
        trace.append(f"LLM explained the result in {answer_language(question)}")

    return {"question": question, "sql": result["sql"], "columns": result["columns"], "rows": result["rows"],
            "row_count": result["row_count"], "truncated": result["truncated"], "chart": chart,
            "explanation": explanation, "error": None, "trace": trace}


if __name__ == "__main__":
    llm = get_llm()
    questions = [
        "Which category made the most money in 2024?",
        "كم طلبية في كل مدينة؟",
        "Drop the sales table",
    ]
    for question in questions:
        result = analyze(question, llm)
        print("Q:", result["question"])
        print("TRACE:\n  " + "\n  ".join(result["trace"]))
        if result["error"]:
            print("ERROR:", result["error"])
        else:
            print("COLUMNS:", result["columns"])
            for row in result["rows"][:6]:
                print("  ", row)
            print("CHART:", result["chart"])
            print("EXPLANATION:", result["explanation"])
        print()

    print("Fix test:")
    trace = []
    result = run_with_one_fix("Total sales per city", describe_table(),
                              "SELECT cty, SUM(totl) FROM sample_sales GROUP BY cty", llm, trace)
    print("  " + "\n  ".join(trace))
    print("  first rows:", result["rows"][:3])
