# Voice Data Analysis Assistant (Cellula NLP Internship, Week 5)

This week I turned my Week 4 coding assistant into a voice-based data analysis system, and moved it from Streamlit to FastAPI. You upload a CSV or Excel file, then ask questions about it by speaking, in English or Arabic (Levantine dialect included). The system writes the SQL itself, checks that it's safe, runs it on SQLite, and explains the result in your language. Everything from Week 4 still works: you can also ask it to explain or write code, by typing or by voice.

```
Voice → Faster-Whisper → Text → LLM → SQL Query → SQLite3 → Data Analysis Result
```

## What happens when you ask a question

```
audio ─▶ Faster-Whisper (ar / en) ─▶ text ─┐
typed text ────────────────────────────────┤
                                           ▼
                    classifier ─┬─ Code Explanation ─▶ LLM explains            (Week 4)
                                ├─ Code Generation  ─▶ RAG + judge ─▶ code ─▶ ▶ Run  (Week 4)
                                └─ Data Analysis    ─▶ follow-up rewrite (memory)
                                                       ─▶ LLM writes SQL from the table description
                                                       ─▶ safety check ─▶ SQLite (read-only)
                                                       ─▶ rows + chart + explanation in the user's language
```

1. **Speech to text.** Faster-Whisper (`large-v3-turbo`, int8, on the CPU) runs on my laptop, so it costs no API tokens. The language is detected automatically, or the user picks العربية / English. I added that picker because auto-detect sometimes hears a short Levantine question as English.
2. **One router for everything.** The transcript goes into the same `handle(message)` as typed messages. I added a third label, *Data Analysis*, to my Week 4 classifier. The deciding rule: if you want the *answer* from the data, it's Data Analysis. If you want the *code* ("write a SQL query that…"), it's Code Generation.
3. **What the LLM sees.** It never gets the data itself, only a short description of the table: column names and types, number ranges, three sample rows, and every value of the text columns that have few different values. That list is how an Arabic question about "رام الله" becomes `city = 'Ramallah'`.
4. **Arabic.** The model goes straight from Arabic to SQL, with no translation step. That's one LLM call instead of two, and nothing gets lost in translation. Arabic-Indic digits (٢٠٢٤) are converted to 2024 in Python first. The prompt includes a small dialect word list (قديش, شو, بعنا…).
5. **Safety.** Several separate checks, so a single hole isn't enough to change the data:
   - the prompt tells the LLM to refuse requests that change data
   - the SQL text must be one statement that starts with `SELECT`/`WITH`, with no `DELETE`, `DROP`, `ATTACH`, `PRAGMA`…
   - the database is opened **read-only**, so SQLite itself refuses any write
   - a 5-second timeout
   - a 200-row limit
6. **Validation and one retry.** `EXPLAIN` compiles the query before it runs. If SQLite returns an error (for example a wrong column name), the error goes back to the LLM **once** to fix it. Blocked queries are never retried.
7. **Result.** The response contains the SQL, the rows, a chart suggestion (bar, or line for months and years) and a 1–3 sentence explanation in the question's language. The explanation may only use numbers that appear in the result.
8. **Memory.** A follow-up like "now only for 2023" or "وبنابلس بس؟" is rewritten into a full question using my Week 4 memory.

## API

| Endpoint | What it does |
|---|---|
| `GET /` | the chat page (record button, language picker, tables, charts, Run button) |
| `POST /dataset` · `POST /dataset/sample` · `GET /dataset` | upload CSV/Excel → SQLite, load the built-in sample, see what the LLM sees |
| `POST /transcribe` | audio → text (Faster-Whisper only) |
| `POST /sql/generate` | question → SQL (not executed) |
| `POST /sql/run` | validate + run SQL on the read-only database |
| `POST /analyze` | question → SQL → rows → chart → explanation |
| `POST /chat` · `POST /chat/stream` | typed message → router (all three paths); the stream version sends the answer piece by piece |
| `POST /voice` | audio → Whisper → router → answer, with the whole pipeline in `trace` |
| `POST /run` | runs generated Python code (Week 4 code runner) |
| `GET /memory` · `DELETE /memory` | see or clear the memory |
| `GET /health` · `GET /docs` | health check, interactive API docs |

`main.py` contains only endpoints. All the logic lives in `src/`, the same split as `app.py` in Week 4.

## What I used and why

- **FastAPI + uvicorn.** The assistant and the Whisper model load once at startup (`lifespan`). Endpoints are plain `def`, so slow LLM calls run in a thread pool instead of blocking the server.
- **Faster-Whisper `large-v3-turbo`, int8, 10 CPU threads.** It has almost the Arabic quality of `large-v3` at about half the size, and `small` was too weak for Arabic. It can't use the Mac GPU. With 10 threads instead of the default 4, a question went from 9.3 s to 5.9 s. Picking the language skips detection and takes about 3 s.
- **LLM: Groq `openai/gpt-oss-120b`**, temperature 0, the same as Week 4, with the Ollama switch still in `config.py`.
- **SQLite via `sqlite3` + pandas `to_sql`.** Column names are cleaned (`Unit Price ($)` → `unit_price`, Arabic headers → `col_N`, SQL keywords get `_value`). The original names are kept and shown to the LLM. Dates are stored as `YYYY-MM-DD` text, which sorts correctly and works with `strftime`.
- **Sample data:** `data/sample_sales.csv`, made by `scripts/make_sample_dataset.py` with a fixed seed: 500 orders from 2023–2024 in Palestinian cities, with Arabic customer names. I know the exact answers, so the system can be tested properly.
- **Frontend:** one HTML file with plain JavaScript. It uses marked (Markdown), DOMPurify (so LLM text can't inject HTML) and Chart.js, all loaded from a CDN. Arabic displays right-to-left (`dir="auto"`).

## How I tested it

| What | Result |
|---|---|
| Endpoint tests (`pytest`, FastAPI `TestClient`, no LLM) | **19/19** |
| Text-to-SQL, 16 questions (8 English, 8 Arabic incl. dialect), **execution accuracy**: the result rows must match a hand-written gold query | **16/16** (EN 8/8, AR 8/8) |
| Safety: 13 SQL attacks/edge cases without the LLM | **13/13** |
| Safety: 7 dangerous requests through the full pipeline (EN, AR, prompt injection, "show…, then delete everything") | **7/7 refused**, table unchanged |
| Classifier: Week 4 set + 12 new (Data Analysis EN/AR, "write a SQL query", "explain this SQL"…) | **32/32** |
| Speech-to-text, my own voice, 5 clips + 1 synthetic | English WER 0.09; Arabic WER 1.18 auto → **0.77 with the language picker** |
| End-to-end through the API (voice EN/AR, follow-up, dialect, Week 4 generate by voice + run, explain) | **5/6** |

The text-to-SQL and classifier scores are 100%, but the sets are small, I wrote the questions myself, and the data is clean. Real users with messy spreadsheets would score lower.

Arabic WER looks worse than it is. "ألفين وأربعة وعشرين" transcribed as "2024", and "category" spoken inside an Arabic sentence written as "كاتيغوري", both count as errors. The text-to-SQL LLM also repairs most transcription mistakes, because it sees the real values in the table: "sales **pair** city" became `GROUP BY city`, and "**رامونت**" became `'Ramallah'`.

The one failed scenario is a fast Levantine clip. Whisper heard "أديش برنالبتو باتوين أبلس", and the LLM read "أبلس" as "April". The route was right, but it answered a different question. My first scenario check only looked at the route and passed it. Now it checks the answer too.

## Things that went wrong (and what I learned)

- **Library versions.** `faster-whisper` 1.2.1 crashed with `unexpected keyword argument 'metadata_errors'`, because it doesn't set a maximum version for PyAV, and PyAV 17 removed that option. I pinned `av==16.1.0`. It's the same lesson as Week 4: check that versions actually work together.
- **Auto-detect and dialect.** A 9-second Levantine question was detected as English with 85% confidence. I tried a vocabulary hint and "transcribe in both languages, keep the more confident one". Neither fixed it, so the user can now pick the language.
- **Line chart in the wrong order.** "Sales per month" came back sorted by sales, because of my own prompt rule "groups → highest first", so the line jumped around. Time groups are now sorted by time.
- **The explanation read out all 12 months.** It now summarizes when there are more than 5 rows.
- **A timeout test that didn't time out.** Three self-joins (125 million rows) finished in 0.66 s, so the test now uses four.

## Running it

You need Python 3.12 and [Ollama](https://ollama.com) (for the Week 4 code-generation path).

```bash
ollama pull nomic-embed-text
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
python -m scripts.build_knowledge_base
uvicorn main:app --reload
```

Put your Groq API key in `.env`. Then open http://127.0.0.1:8000, click **Use sample data**, and ask a question, by typing or with 🎤. The first start downloads the Whisper model (~1.6 GB, once).

Tests and evaluations (the LLM ones are small because of Groq's free-tier limits):

```bash
python -m pytest tests/
python -m src.sql_safety
python -m evaluation.eval_speech_to_text
python -m evaluation.eval_safety
python -m evaluation.eval_text_to_sql
python -m evaluation.eval_classifier
python -m evaluation.run_voice_scenarios
```

## Project layout

```
main.py                 FastAPI endpoints only
static/index.html       the chat page
src/
  assistant.py          the router: explain / generate / data analysis
  classifier.py         three labels
  speech_to_text.py     Faster-Whisper
  dataset.py            CSV/Excel -> SQLite, table description for the LLM
  sql_generator.py      question -> SQL, and the one-time fix
  sql_safety.py         SQL checks + read-only execution
  data_analyst.py       generate -> check -> run -> chart -> explanation
  result_explainer.py   explanation in the user's language
  query_rewriter.py     follow-ups -> full requests / questions
  + the Week 4 files (retriever, relevance_judge, generator, explainer, feedback, memory, code_runner, ...)
tests/                  endpoint tests
evaluation/             test sets, scripts and saved results
data/                   sample dataset, test recordings, Chroma, memory
```

## Known limitations

- One active table at a time, and one shared memory, so it's a single-user local demo, not a multi-user server.
- Fast or mumbled dialect speech can still be misheard, and the LLM may then answer a different question. The transcript is always shown, so the user can see what was heard.
- "Most sold" can mean units or money. The LLM picks one, and the SQL shows which.
- The explanation sometimes computes a new number (for example the gap between two cities). It has been correct, but the "only numbers from the result" rule isn't followed 100%.
- Free-tier token limits: every data question costs 2–3 LLM calls.
