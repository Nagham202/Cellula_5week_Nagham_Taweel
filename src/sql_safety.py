import re
import sqlite3
import time
from contextlib import closing

from src.config import MAX_RESULT_ROWS, SQL_TIMEOUT_SECONDS
from src.dataset import connect_read_only

# any of these means the query wants to change something
FORBIDDEN_WORDS = ["insert", "update", "delete", "drop", "alter", "create", "attach", "detach",
                   "pragma", "vacuum", "reindex", "analyze", "truncate", "grant"]


class UnsafeSQLError(Exception):
    """Raised when a query is blocked before running."""


def code_without_strings_and_comments(sql):
    # blank out text values first so a name like 'Drop' doesn't trigger the check
    sql = re.sub(r"'(?:[^']|'')*'", "''", sql)
    sql = re.sub(r"--[^\n]*", " ", sql)
    sql = re.sub(r"/\*.*?\*/", " ", sql, flags=re.DOTALL)
    return sql.strip().rstrip(";").strip().lower()


def check_sql(sql):
    """Returns the cleaned query, or raises UnsafeSQLError."""
    sql = sql.strip().rstrip(";").strip()
    code = code_without_strings_and_comments(sql)
    if not code:
        raise UnsafeSQLError("The query is empty.")
    # "SELECT 1; DROP TABLE x" is two statements hiding behind a select
    if ";" in code:
        raise UnsafeSQLError("Only one SQL statement is allowed.")
    if not re.match(r"(select|with)\b", code):
        raise UnsafeSQLError("Only SELECT queries are allowed (the data can only be read).")
    # still needed after the select check, sqlite allows WITH ... DELETE
    for word in FORBIDDEN_WORDS:
        if re.search(rf"\b{word}\b", code):
            raise UnsafeSQLError(f"{word.upper()} is not allowed (the data can only be read).")
    # replace() is also a normal text function, so only REPLACE INTO is blocked
    if re.search(r"\breplace\s+into\b", code):
        raise UnsafeSQLError("REPLACE INTO is not allowed (the data can only be read).")
    return sql


def run_select(sql, max_rows=MAX_RESULT_ROWS, timeout=SQL_TIMEOUT_SECONDS):
    sql = check_sql(sql)
    start = time.time()
    deadline = time.monotonic() + timeout

    with closing(connect_read_only()) as connection:
        # sqlite calls this every 10k steps, returning True stops the query
        connection.set_progress_handler(lambda: time.monotonic() > deadline, 10_000)
        try:
            # EXPLAIN catches bad columns and syntax without reading any data
            connection.execute(f"EXPLAIN {sql}")
            cursor = connection.execute(sql)
            columns = [description[0] for description in cursor.description]
            # the extra row tells us the result was cut (safer than editing the sql to add LIMIT)
            rows = cursor.fetchmany(max_rows + 1)
        except sqlite3.OperationalError as error:
            if "interrupted" in str(error):
                raise sqlite3.OperationalError(f"Stopped: the query ran longer than {timeout} seconds.")
            raise

    return {
        "sql": sql,
        "columns": columns,
        "rows": [list(row) for row in rows[:max_rows]],
        "row_count": min(len(rows), max_rows),
        "truncated": len(rows) > max_rows,
        "seconds": round(time.time() - start, 3),
    }


if __name__ == "__main__":
    # hand written queries, no llm, so this costs no tokens
    tests = {
        "1. normal query": "SELECT city, COUNT(*) AS orders FROM sample_sales GROUP BY city ORDER BY orders DESC;",
        "2. text value with & inside": "SELECT COUNT(*) FROM sample_sales WHERE category = 'Home & Kitchen'",
        "3. DELETE": "DELETE FROM sample_sales",
        "4. DROP TABLE": "DROP TABLE sample_sales",
        "5. two statements": "SELECT 1; DROP TABLE sample_sales",
        "6. DELETE hidden in WITH": "WITH x AS (SELECT 1) DELETE FROM sample_sales",
        "7. wrong column name": "SELECT totl FROM sample_sales",
        "8. big result (cut)": "SELECT * FROM sample_sales",
        "9. very slow query": "SELECT COUNT(*) FROM sample_sales a, sample_sales b, sample_sales c, sample_sales d",
    }
    for name, sql in tests.items():
        try:
            result = run_select(sql)
            print(f"✅ {name}: {result['row_count']} rows, truncated={result['truncated']}, "
                  f"{result['seconds']}s, first row = {result['rows'][0]}")
        except UnsafeSQLError as error:
            print(f"🛑 {name}: BLOCKED -> {error}")
        except sqlite3.Error as error:
            print(f"⚠️  {name}: SQLite error -> {error}")

    with closing(connect_read_only()) as connection:
        try:
            connection.execute("DELETE FROM sample_sales")
        except sqlite3.OperationalError as error:
            print(f"🛑 10. DELETE that skips the text check: SQLite refused -> {error}")
