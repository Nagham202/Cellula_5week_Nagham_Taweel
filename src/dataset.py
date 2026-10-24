import io
import json
import re
import sqlite3
from contextlib import closing
from pathlib import Path

import pandas as pd

from src.config import DATASET_DB_PATH, DATASET_INFO_PATH, DISTINCT_VALUES_LIMIT, SAMPLE_ROW_COUNT

# a column called order or group would break the queries, so these get renamed
SQL_KEYWORDS = {"select", "from", "where", "group", "order", "by", "limit", "table", "index", "join", "on",
                "and", "or", "not", "as", "case", "when", "values", "count", "sum", "date", "time", "key"}


class DatasetError(Exception):
    """Something wrong with the uploaded file (wrong type, empty...)."""


def clean_name(name, fallback):
    text = re.sub(r"[^a-z0-9]+", "_", str(name).strip().lower()).strip("_")
    # arabic headers end up empty here, the original name is still shown to the llm
    if not text:
        return fallback
    if text[0].isdigit():
        text = f"col_{text}"
    if text in SQL_KEYWORDS:
        text = f"{text}_value"
    return text


def clean_column_names(columns):
    clean = []
    for position, name in enumerate(columns, start=1):
        base = clean_name(name, f"col_{position}")
        new_name, number = base, 2
        # two headers can clean to the same name ("Price" and "price ")
        while new_name in clean:
            new_name, number = f"{base}_{number}", number + 1
        clean.append(new_name)
    return clean


def read_file(content, file_name):
    suffix = Path(file_name).suffix.lower()
    if suffix == ".csv":
        # arabic csv files saved from excel on windows are often cp1256
        for encoding in ("utf-8-sig", "cp1256"):
            try:
                return pd.read_csv(io.BytesIO(content), encoding=encoding)
            except UnicodeDecodeError:
                continue
        raise DatasetError("Could not read the CSV text encoding (try saving it as UTF-8).")
    if suffix in (".xlsx", ".xls"):
        return pd.read_excel(io.BytesIO(content))
    raise DatasetError("Only .csv and .xlsx files are supported.")


# sqlite has no date type, YYYY-MM-DD text sorts right and works with strftime
def normalize_dates(df):
    for column in df.columns:
        if "date" in column and not pd.api.types.is_numeric_dtype(df[column]):
            parsed = pd.to_datetime(df[column], errors="coerce")
            # only convert if almost everything parsed, otherwise it's probably not a date
            if parsed.notna().mean() > 0.9:
                df[column] = parsed.dt.strftime("%Y-%m-%d")
    return df


def load_dataset(content, file_name):
    """Saves an uploaded CSV/Excel file as the active SQLite table."""
    try:
        df = read_file(content, file_name)
    except DatasetError:
        raise
    except Exception as error:
        raise DatasetError(f"Could not read the file: {error}")

    df = df.dropna(how="all")
    if df.empty:
        raise DatasetError("The file has no data rows.")

    original_columns = [str(column) for column in df.columns]
    df.columns = clean_column_names(original_columns)
    df = normalize_dates(df)
    table = clean_name(Path(file_name).stem, "data")

    DATASET_DB_PATH.parent.mkdir(exist_ok=True)
    # one table at a time, a new upload replaces the old one
    DATASET_DB_PATH.unlink(missing_ok=True)
    # closing() actually closes the file, a plain with only commits
    with closing(sqlite3.connect(DATASET_DB_PATH)) as connection:
        df.to_sql(table, connection, index=False)

    info = {"table": table, "file_name": file_name, "columns": dict(zip(df.columns, original_columns))}
    DATASET_INFO_PATH.write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    return {**info, "rows": len(df)}


def load_info():
    if not DATASET_DB_PATH.exists() or not DATASET_INFO_PATH.exists():
        return None
    return json.loads(DATASET_INFO_PATH.read_text(encoding="utf-8"))


def dataset_summary():
    """One line about the loaded table, for the classifier."""
    info = load_info()
    if info is None:
        return "Loaded dataset: none."
    return f"Loaded dataset: table {info['table']} with columns {', '.join(info['columns'])}."


def connect_read_only():
    # mode=ro means sqlite itself refuses any write
    return sqlite3.connect(f"file:{DATASET_DB_PATH}?mode=ro", uri=True)


def describe_column(cursor, table, name, sql_type, original):
    line = f'- {name} {sql_type}'
    if original != name:
        line += f' (original name: "{original}")'

    if sql_type in ("INTEGER", "REAL"):
        low, high = cursor.execute(f'SELECT MIN("{name}"), MAX("{name}") FROM "{table}"').fetchone()
        return f"{line}, range {low} to {high}"

    distinct = cursor.execute(f'SELECT COUNT(DISTINCT "{name}") FROM "{table}"').fetchone()[0]
    if "date" in name:
        low, high = cursor.execute(f'SELECT MIN("{name}"), MAX("{name}") FROM "{table}"').fetchone()
        return f"{line}, format YYYY-MM-DD, range {low} to {high}"
    # listing all the values lets the llm match "رام الله" to 'Ramallah'
    if distinct <= DISTINCT_VALUES_LIMIT:
        values = [row[0] for row in cursor.execute(
            f'SELECT DISTINCT "{name}" FROM "{table}" WHERE "{name}" IS NOT NULL ORDER BY 1')]
        return f"{line}, all values: {', '.join(repr(value) for value in values)}"
    examples = [row[0] for row in cursor.execute(f'SELECT DISTINCT "{name}" FROM "{table}" LIMIT 5')]
    return f"{line}, {distinct} different values, e.g. {', '.join(repr(value) for value in examples)}"


def describe_table():
    """Describes the table for the LLM: columns, values and a few sample rows."""
    info = load_info()
    if info is None:
        return None
    table = info["table"]

    with closing(connect_read_only()) as connection:
        cursor = connection.cursor()
        row_count = cursor.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0]
        columns = [(row[1], row[2]) for row in cursor.execute(f'PRAGMA table_info("{table}")')]

        lines = [f"Table: {table} ({row_count} rows)", "Columns:"]
        for name, sql_type in columns:
            lines.append(describe_column(cursor, table, name, sql_type, info["columns"].get(name, name)))

        sample = cursor.execute(f'SELECT * FROM "{table}" LIMIT {SAMPLE_ROW_COUNT}').fetchall()

    lines.append("Sample rows:")
    lines.append(" | ".join(name for name, _ in columns))
    lines.extend(" | ".join(str(value) for value in row) for row in sample)
    return "\n".join(lines)
