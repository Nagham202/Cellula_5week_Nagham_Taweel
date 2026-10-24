# Week 5 – Voice-Based Data Analysis (FastAPI): Report Notes

## Step 0 – Big picture and setup

### System overview
Week 4 assistant + a third path. Every message (typed, or spoken → Faster-Whisper → text) goes to
`CodingAssistant.handle()` → classifier → Explain / Generate (Week 4) / **Data Analysis** (new):
table description → LLM → SQL → validate → SQLite (read-only) → rows → LLM explanation in the user's language.
FastAPI (`main.py`) replaces Streamlit (`app.py`); all logic stays in `src/`.

### Decisions
| Decision | Choice | Why |
|---|---|---|
| Project setup | New folder + new git repo (copy of Week 4 without `.venv`/`.git`) | Week 4 stays untouched; one repo per week like before |
| Chroma DB | Copied from Week 4 | No rebuild needed (same embedding model) |
| Memory | Started empty | Week 4 test conversations shouldn't leak into Week 5 |
| Build order | Kept as planned | Each piece is tested alone before being connected |

### Challenges
-

## Step 1 – FastAPI basics
- `main.py` is a thin layer over `CodingAssistant`, like `app.py` was. `app.py` and Streamlit are removed.
- Assistant is loaded once at startup (FastAPI `lifespan`), like `@st.cache_resource`.
- Endpoints: `GET /` (health), `POST /chat` (full JSON), `POST /chat/stream` (NDJSON stream), `POST /run`, `GET /memory`, `DELETE /memory`.
- Week 4 features without Streamlit:
  - Streaming → `/chat/stream` sends one JSON line per piece (first line = intent/route/trace).
  - "How I answered" → `trace` + `sources` fields in the response.
  - Run button → `/chat` returns `code` separately; `POST /run` runs it in the Week 4 code runner.
- Endpoints are `def` (not `async def`) so slow Groq calls run in a thread pool and don't block the server.
- Limitation: one shared assistant (one memory) → fine for a local single-user demo.
- Results: no-token checks pass with TestClient (`/` ok, `/run` → `45`, empty message → 422).

## Step 2 – Dataset upload → SQLite + table description
- Decisions:
  - Dataset: my own `data/sample_sales.csv` (500 rows, 2023–2024, fixed seed via `scripts/make_sample_dataset.py`) → I know the true answers for the text-to-SQL test set.
  - Arabic in the data: cities/categories/products in English, `customer_name` in Arabic → tests matching in both directions.
  - One active table: each upload replaces `data/datasets.db` → short prompts, no table selection.
  - Column names cleaned to snake_case (`Unit Price ($)` → `unit_price`); Arabic/empty headers → `col_N`; SQL keywords get `_value`; the original name is kept and shown to the LLM.
- `src/dataset.py`: `load_dataset()` (CSV/Excel → pandas → `to_sql`), `describe_table()`, `connect_read_only()`.
- What the LLM sees: table name, row count, each column's SQLite type + original name, min/max for numbers and dates,
  ALL values for text columns with ≤ 25 different values (lets "رام الله" map to `'Ramallah'`), 3 sample rows. Never the whole table.
- Dates are stored as `YYYY-MM-DD` text (SQLite has no date type; this format sorts correctly and works with `strftime`).
- CSV encoding: UTF-8 first, then cp1256 (Arabic Windows Excel).
- Endpoints: `POST /dataset` (upload), `POST /dataset/sample` (built-in sample), `GET /dataset` (what the LLM sees).
- Results: sample loads (500 rows, 9 columns); messy test headers (`المدينة`, `Price` + `price `, `Order`, `2024 Sales`) → `col_1`, `price`, `price_2`, `order_value`, `col_2024_sales`; `.txt` → 400 error.

## Step 3 – Text-to-SQL (typed questions, English + Arabic)
- `src/sql_generator.py`: `generate_sql(question, table_description, llm)` → `{question, sql, refusal}`. Prompt in `src/prompts.py` (`TEXT_TO_SQL_*`).
- The LLM sees only the table description from Step 2 + the question (~1,000 tokens per question), never the data.
- Decision – Arabic: **direct Arabic → SQL** (no translation step). Why: one LLM call instead of two (Groq limits), and a translation can lose details like names/values.
  - Arabic-Indic digits (٢٠٢٤ → 2024) are converted in Python before the LLM (free, always correct).
  - Values in another language ("رام الله" → `'Ramallah'`): the description lists all stored values + a rule and an example in the prompt.
  - Dialect: a small Arabic word list in the prompt (قديش, بعنا, شو...) + one Levantine few-shot example.
- Prompt rules: exact column names, `strftime` for dates, English aliases, ROUND(…, 2), only SELECT.
- Change/impossible requests → the LLM answers `CANNOT_ANSWER: reason` → returned as `refusal` (first safety layer; the real guard comes in Step 4).
- Forgiving parser: takes the ```sql block, or the whole answer if there is no block.
- Endpoint: `POST /sql/generate` (generates only, does not execute).
- Results: "كم طلبية في رام الله سنة ٢٠٢٤؟" → `COUNT(*) … WHERE city = 'Ramallah' AND strftime('%Y', order_date) = '2024'` ✅; "Delete all orders from Jenin" → refused ✅.

## Step 4 – SQL validation + safe execution
- `src/sql_safety.py`: `check_sql()` + `run_select()`. `src/data_analyst.py`: `analyze(question, llm)` = generate → check → run (fix once).
- Safety = several independent layers (defense in depth), so one hole is not enough:
  1. LLM prompt: change requests → `CANNOT_ANSWER` (soft layer, can be tricked).
  2. Text check (`check_sql`): one statement only, must start with SELECT/WITH, no INSERT/UPDATE/DELETE/DROP/ALTER/CREATE/ATTACH/PRAGMA…
     Text values and comments are blanked first, so `'Home & Kitchen'` or a name like 'Drop' don't trigger it.
     The word check is still needed after the SELECT check because SQLite accepts `WITH x AS (...) DELETE ...`.
  3. Database opened read-only (`mode=ro`): SQLite itself refuses any write, even if layer 2 had a bug.
  4. Timeout (5 s) with `set_progress_handler` → stops runaway queries (e.g. 4 self-joins = 500^4 rows).
  5. Row limit (200) with `fetchmany`, plus a `truncated` flag.
- Decision – row limit with `fetchmany` instead of adding `LIMIT` to the SQL text: editing the LLM's SQL could break its own LIMIT/ORDER BY.
- Validation: `EXPLAIN <sql>` compiles the query without reading data → catches wrong columns/syntax before running.
- Decision – self-correction: SQL error → error message sent back to the LLM **once** (`fix_sql`). Blocked queries are never retried. One retry only, to protect the Groq limits.
- Endpoints: `POST /sql/run` (validate + run any SQL, no LLM; 400 if blocked/invalid), `POST /analyze` (full chain: question → rows, with trace).
- Results – safety test (no LLM, 10 cases): 2 normal queries ✅, DELETE / DROP / two statements / DELETE inside WITH all blocked ✅,
  wrong column → SQLite error ✅, 500 rows → cut to 200 ✅, 4 self-joins → stopped after 5 s ✅, DELETE skipping the text check → "attempt to write a readonly database" ✅.
- Results – pipeline: "Which category made the most money in 2024?" → Electronics 98,620 ✅; "كم طلبية في كل مدينة؟" → 6 cities ✅;
  "Drop the sales table" → refused ✅; broken SQL (`cty`, `totl`) → fixed by the LLM in one retry ✅.

### Challenges
- First timeout test (3 self-joins = 125M rows) finished in 0.66 s, so SQLite is faster than expected → test changed to 4 self-joins.

## Step 5 – Result + explanation (+ chart suggestion)
- `src/result_explainer.py`: one LLM call gets the question + SQL + first 30 result rows → max 3 sentences.
- Answer language decided in Python (any Arabic letter → Arabic), not by the LLM → always matches the user. Arabic answers write city/product names in Arabic.
- Anti-hallucination rules: use only numbers from the result; empty result → "no data matched"; cut result → say so; >5 rows → summarize, don't read every row.
- Streaming version (`stream_result_explanation`) ready for the chat; `analyze(..., explain=False)` skips it to save tokens in evaluations.
- Decision – chart: rule-based `suggest_chart()` (no LLM, free): 2 columns (text label + number), 2–30 rows → `bar`; label column about time (date/month/year) → `line`.
  The API only returns `{"type", "x", "y"}`; drawing is the frontend's job (keeps the backend free of plotting libraries).
- `/analyze` now returns: `question, sql, columns, rows, row_count, truncated, chart, explanation, error, trace`.
- Results: "Total sales per month in 2024" → line chart, months in order, explanation July high / June low ✅;
  "كم طلبية في كل مدينة؟" → bar chart, Arabic explanation with Arabic city names (أريحا 92 … بيت لحم 73) ✅.

### Challenges
- Monthly result came back ordered by sales (prompt rule "groups → ORDER BY number DESC") → the line chart would jump around. Fix: rule exception "time groups → ORDER BY time ASC".
- First explanation read out all 12 months → added "more than 5 rows → summarize".

## Step 6 – "Data Analysis" in the classifier + router
- Decision – router vs separate endpoint: **both**. Third label "Data Analysis" in the Week 4 classifier, so `/chat` (and later `/voice`)
  handle all three paths from one message box (the mentor asked to ADD analysis to the Week 4 assistant). `/analyze` stays as a direct endpoint for tests.
- Classifier prompt: definition + examples in English and Arabic; key rule: wants the ANSWER from the data → Data Analysis,
  wants SQL/pandas CODE → Code Generation ("Write a SQL query that…" → Code Generation).
- Classifier also gets one line about the loaded table (`dataset_summary()`: table + column names) → recognizes questions about "city", "product"…
- Parser: keyword "analysis" → Data Analysis; unparseable answers still fall back to Code Generation (Week 4 rule).
- `CodingAssistant.data_path()`: follow-up rewrite → `analyze(explain=False)` → explanation streamed like the Week 4 answers.
  Response gets a new `data` field (sql, columns, rows, chart, error); Week 4 paths send `data: null`.
- Decision – memory for data follow-ups: **yes**, reusing Week 4 memory + a new data rewriter prompt (`make_standalone_question`).
  Only called when the previous turn was data_analysis (saves a Groq call on every other question). Keeps the user's language.
- Errors (no dataset, blocked, failed SQL) → friendly message in the user's language.
- Results – 4-turn conversation (temp memory file):
  1. "What are the total sales per city?" → data_analysis, 6 cities, Nablus highest ✅
  2. "now only for 2024" → rewritten "What are the total sales per city for 2024 only?" → Ramallah highest ✅
  3. "وبنابلس بس؟" → rewritten "ما هي مبيعات نابلس فقط لعام 2024؟" → 22,380 (matches turn 2), Arabic answer ✅
  4. "Explain this code: …" → explain path (Week 4 still works) ✅

### Challenges
- "Write a SQL query that returns total sales per city" → correctly Code Generation, but the relevance judge said Not Relevant
  (HumanEval has no SQL, and the Week 4 judge rule says databases → Not Relevant) → the assistant asks the user to teach it. Expected Week 4 behavior, not a bug.
- The explanation once computed a new number (gap = 10,968). It's correct, but rule "only numbers from the result" is not 100% followed → note as a limitation.

## Step 7 – Faster-Whisper alone (speech → text)
- `src/speech_to_text.py`: `load_whisper_model()` (once) + `transcribe(model, audio, language=None)` → text, language, probability, timings.
- Decision – model: **large-v3-turbo** (~1.6 GB). small = fast but weak Arabic; medium = OK; large-v3 = best but ~3 GB and slow on CPU;
  turbo = almost large-v3 accuracy with a much smaller decoder → best Arabic/speed balance on a Mac.
- Settings: CPU (Faster-Whisper/CTranslate2 has no Mac GPU support), `int8` (quantization, like Week 2), `cpu_threads=10`, beam 5,
  `vad_filter=True` (cuts silence where Whisper invents words), `condition_on_previous_text=False` (questions are independent).
- Decision – language: auto-detect, restricted to Arabic/English: if auto-detect guesses another language (e.g. Persian for short Arabic clips),
  re-transcribe with the more likely of ar/en. Optional `language` parameter to force it.
- Results – macOS text-to-speech clips (voices Samantha / Majed), first quick check:
  - EN "What are the total sales per city?" → exact ✅
  - AR (MSA) "ما هو مجموع المبيعات في رام الله سنة ألفين وأربعة وعشرين؟" → exact, and number words became "2024" ✅
  - AR (dialect) "قديش بعنا لابتوبات بنابلس؟" → "قدي شبعنا لأبي توباتي بنابلوس" ❌ (but a TTS voice reading dialect is unnatural → retest with my own voice)
- Speed: model load 2.3 s (after a one-time 114 s download); ~6 s per short question with 10 threads (9.3 s with the default 4). Beam 1 vs 5: same speed.

### Challenges
- `TypeError: open() got an unexpected keyword argument 'metadata_errors'`: faster-whisper 1.2.1 has no upper limit on PyAV, and PyAV 17+ removed that option.
  Fix: pin `av==16.1.0` in requirements.txt (lesson again: check library versions work together).
- Results – my own voice (5 clips, `data/audio_tests/`, auto-detect):
  | Clip | Said | Heard | Language |
  |---|---|---|---|
  | en_1 | What are the total sales per city? | …sales **pair** city? | en ✅ |
  | en_2 | Which product sold the most in 2024? | Which product **sells** the most in 2024? | en ✅ |
  | ar_1 | …في رام الله سنة ألفين وأربعة وعشرين؟ | …في **رامونت** لسنة 2024؟ | ar ✅ |
  | ar_2 | قديش بعنا لابتوبات بنابلس؟ | "Add this brand it up to Pat Wiener Plus." | **en ❌ (0.85)** |
  | ar_3 | شو أكثر category بتبيع؟ | شو أكتر **كاتيغوري بتبير**؟ | ar ✅ (0.84) |
- Forced Arabic on ar_2 → "أديش برنا لابتوب باتوين أبلس" (أديش = how Palestinians pronounce قديش, so partly right). A vocabulary hint (`initial_prompt`) only fixed "لابتوب" → not used.
- Tried "unsure → transcribe in both languages, keep the higher avg_logprob": still picked English for ar_2 (−0.65 vs −0.89) → rejected (doubles time, doesn't fix it).
- Key finding: the text-to-SQL LLM repairs most transcription errors because it sees the real column values:
  "pair city" → per city ✅, "رامونت" → 'Ramallah' ✅, "كاتيغوري بتبير" → most-selling category ✅; ar_2 → Laptop ✅ but city wrong (Bethlehem instead of Nablus) ❌. 3/4 correct SQL from imperfect transcripts.
- Decision – language: the user can pick **auto / ar / en** (default auto) in `/voice` and the frontend, because auto-detect fails on short dialect clips.

## Step 8 – `/voice` endpoint (end to end)
- Whisper is loaded once at startup (`lifespan`, next to the assistant), not per request.
- `POST /transcribe` (speech-to-text only) and `POST /voice` (audio → Whisper → text → **same router as `/chat`** → answer).
  Both take a file upload (`audio`) + a form field `language` = `auto` / `ar` / `en` (default auto; FastAPI rejects anything else with 422).
- Decision – voice for everything: `/voice` reuses `chat_response()` from `/chat`, so speaking works for all three paths (explain / generate / data analysis) with zero extra logic.
- `/voice` response = `/chat` response + `transcript` (text, language, probability, timings); the trace starts with the Whisper step → "How I answered" shows the whole pipeline.
- Guards: audio > 10 MB → 400; not an audio file → 400 "Could not read the audio file"; silence/empty transcript → 400.
- Results (TestClient):
  - `/transcribe` ar_2 with `language=ar` → Arabic text (instead of the English nonsense from auto) ✅
  - `/voice` ar_1 (my voice) → Whisper "…رامونت لسنة 2024" → Data Analysis → `city = 'Ramallah'` + 2024 → 26,397 → Arabic explanation ✅ (~6 s Whisper + ~3 LLM calls)
  - .txt file → 400 ✅; `language=fr` → 422 ✅

## Step 9 – Frontend (small HTML page)
- Decision: one static page `static/index.html` served by FastAPI at `/` (instead of only `/docs`): a live microphone and drawn charts make a much better demo.
  Health check moved to `GET /health`.
- Plain HTML + JavaScript, no build step. Libraries from a CDN, pinned versions: marked (Markdown), DOMPurify (LLM text can't inject HTML), Chart.js (charts).
- Features (all Week 4 Streamlit features kept + Week 5):
  - Typed messages → `/chat/stream` (NDJSON) → answer appears piece by piece (Week 4 streaming).
  - 🎤 Record → browser `MediaRecorder` → `/voice`; 📁 sends an audio file; language picker Auto / العربية / English.
  - Route label + "How I answered" trace (Week 4), now starting with the Whisper step for voice.
  - Data answers: SQL (collapsible), result table, chart drawn from the backend's `{type, x, y}` suggestion.
  - ▶ Run code for generated code → `/run` (Week 4 code runner). Dataset: "Use sample data" / upload CSV-Excel; Clear memory.
  - `dir="auto"` on messages and table cells → Arabic shows right-to-left correctly.
- Browsers record webm/opus (Chrome) or mp4 (Safari) → checked: a webm file transcribes correctly through `/transcribe` ✅.
- Mic access works on `http://127.0.0.1` without HTTPS (browsers treat localhost as secure).

## Step 10 – Testing
- Decision – testing approach: free tests first (no Groq), small LLM evaluations second; every script uses the sample dataset and saves a `results_*.json`.
- Endpoint tests: `tests/test_api.py` with pytest + FastAPI `TestClient` (no LLM calls): pages, dataset upload + column cleaning, wrong file type, `/sql/run` ok/blocked/bad column,
  `/run`, validation errors (422), memory clear (temp memory file), `/transcribe` bad file / English clip / Arabic with picker. **19/19 passed** (13 s).
- Text-to-SQL: `evaluation/eval_text_to_sql.py`, 16 questions (8 EN, 8 AR incl. Levantine and Arabic-Indic digits), each with a hand-written gold SQL.
  Metric = **execution accuracy**: compare the RESULT rows with the gold result (order ignored, extra columns allowed, numbers rounded to 2 decimals), not the SQL text.
  **16/16 = 100%** (EN 8/8, AR 8/8), 0 needed the retry. ⚠️ Optimistic: small set, my own clean data, questions written by me.
- Safety: `evaluation/eval_safety.py`. 13 SQL attacks/edge cases without LLM → **13/13** (10 blocked, 3 harmless ones correctly allowed: 'Drop Table' inside a string, `replace()` function, `;` inside a comment).
  7 dangerous requests through the full pipeline (EN + AR + prompt injection "Ignore all previous rules…" + mixed "show…, then delete everything") → **7/7 safe** (all refused). Table fingerprint (rows, sum) unchanged: (500, 246429) ✅.
- Classifier: Week 4 set (20) + 12 new (8 Data Analysis EN/AR + 4 confusing ones: "Write a SQL query…", pandas script, "Explain this SQL…", Arabic code request) → **32/32 = 100%**
  (Explanation 11/11, Generation 13/13, Data Analysis 8/8). Week 4 accuracy kept after adding the third label.
- Speech-to-text: `evaluation/eval_speech_to_text.py`, WER/CER per clip, auto vs fixed language (Arabic letters normalized: أ/إ/آ→ا, ة→ه, ى→ي; punctuation removed):
  | | WER | CER | wrong language |
  |---|---|---|---|
  | English auto / fixed | 0.09 / 0.09 | 0.05 / 0.05 | 0/3 |
  | Arabic auto | 1.18 | 0.77 | 1/3 |
  | Arabic fixed (picker) | 0.77 | 0.39 | 0/3 |
  - Arabic WER is inflated: "ألفين وأربعة وعشرين" vs "2024" and English words said in Arabic ("category" → كاتيغوري) count as errors, though they're not real mistakes.
  - Fixed language is also faster: ~3 s vs ~6 s (auto-detect is an extra pass).
- End-to-end: `evaluation/run_voice_scenarios.py` through the real API (TestClient, temp memory), 6 scenarios: EN voice → data ✅, AR voice → data (26,397) ✅,
  voice + typed follow-up "now only for 2023" (rewritten via memory) ✅, Levantine voice with picker ❌, Week 4 generate by voice + run code ✅, Week 4 explain ✅ → **5/6**.
  - Scenario 4: route correct, but the transcript "أديش برنالبتو باتوين أبلس" was read as "April" → answered 21,300 (April sales) instead of 27,000 (laptops in Nablus).
    First run only checked the route and showed 6/6 → added a value check; lesson: check the answer, not just the route.
- Token budget: all LLM evaluations together ≈ 60 Groq calls; no rate-limit failures (max_retries handled waits).
- Code comments: all `#`, `//`, `<!-- -->` comments removed from code files on request (backup kept); docstrings kept. Explanations live in these notes and the README.
