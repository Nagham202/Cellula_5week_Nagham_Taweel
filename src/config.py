import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"

load_dotenv(PROJECT_DIR / ".env")

# switch to "ollama" to run everything locally (slower, but no limits)
LLM_PROVIDER = "groq"

# llama-3.3-70b wasn't available on my account
GROQ_MODEL = "openai/gpt-oss-120b"
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

OLLAMA_MODEL = "llama3.1:8b"

# 0 so the classifier and the judge give the same answer every time
LLM_TEMPERATURE = 0

# if this changes, rebuild the chroma db
EMBEDDING_MODEL = "nomic-embed-text"

CHROMA_DIR = DATA_DIR / "chroma_db"
COLLECTION_NAME = "coding_knowledge"

HELD_OUT_PATH = DATA_DIR / "held_out_tasks.json"
HELD_OUT_COUNT = 20
# fixed seed so the same 20 problems are held out every time
RANDOM_SEED = 42

CHUNK_SIZE = 2500
CHUNK_OVERLAP = 200

TOP_K = 3

# stops infinite loops
RUN_TIMEOUT_SECONDS = 10
MAX_OUTPUT_CHARS = 10_000

USER_SOLUTIONS_PATH = DATA_DIR / "user_solutions.jsonl"

MEMORY_PATH = DATA_DIR / "memory.json"
# everything is saved but only this much goes to the llm (free tier is 8k tokens/min)
MEMORY_TURNS_IN_PROMPT = 3
MEMORY_TEXT_LIMIT = 400
MEMORY_CODE_LIMIT = 2000
MAX_FOLLOW_UP_CHARS = 200

DATASET_DB_PATH = DATA_DIR / "datasets.db"
DATASET_INFO_PATH = DATA_DIR / "dataset_info.json"
SAMPLE_DATASET_PATH = DATA_DIR / "sample_sales.csv"
# text columns with this many values or less get all of them listed for the llm
DISTINCT_VALUES_LIMIT = 25
SAMPLE_ROW_COUNT = 3
MAX_RESULT_ROWS = 200
# kills runaway queries, e.g. a table joined with itself 4 times
SQL_TIMEOUT_SECONDS = 5
EXPLAIN_ROWS_LIMIT = 30
CHART_MAX_POINTS = 30

# turbo is almost as good as large-v3 for arabic but a lot faster
WHISPER_MODEL = "large-v3-turbo"
# faster-whisper can't use the mac gpu
WHISPER_DEVICE = "cpu"
# int8 = quantized, faster with almost the same accuracy
WHISPER_COMPUTE_TYPE = "int8"
WHISPER_BEAM_SIZE = 5
# default is 4, with 10 a question went from ~9s to ~6s
WHISPER_CPU_THREADS = 10
WHISPER_LANGUAGES = ("ar", "en")
AUDIO_TESTS_DIR = DATA_DIR / "audio_tests"
MAX_AUDIO_MB = 10
