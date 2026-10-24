import json
import sqlite3
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field

from src.assistant import CodingAssistant
from src.code_runner import extract_python_code, run_code
from src.config import MAX_AUDIO_MB, SAMPLE_DATASET_PATH
from src.dataset import DatasetError, describe_table, load_dataset
from src.data_analyst import analyze
from src.sql_generator import generate_sql
from src.speech_to_text import load_whisper_model, transcribe
from src.sql_safety import UnsafeSQLError, run_select


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, examples=["Write a Python function that reverses a string"])


class QuestionRequest(BaseModel):
    question: str = Field(min_length=1, examples=["What are the total sales per city?"])


class SqlRequest(BaseModel):
    sql: str = Field(min_length=1, examples=["SELECT city, COUNT(*) AS orders FROM sample_sales GROUP BY city"])


class RunRequest(BaseModel):
    code: str = Field(min_length=1, examples=["print(sum(range(10)))"])


# load the assistant and whisper once at startup, both are slow to load
@asynccontextmanager
async def lifespan(app):
    app.state.assistant = CodingAssistant()
    app.state.whisper = load_whisper_model()
    yield


app = FastAPI(title="Voice Data Analysis Assistant", lifespan=lifespan)


def sources_as_json(sources):
    return [{"doc_id": doc.metadata["doc_id"], "distance": round(distance, 3)} for doc, distance in sources]


def answer_info(result):
    return {
        "intent": result["intent"],
        "route": result["route"],
        "trace": result["trace"],
        "sources": sources_as_json(result["sources"]),
        "data": result.get("data"),
    }


def json_line(data):
    # ensure_ascii=False keeps arabic readable
    return json.dumps(data, ensure_ascii=False) + "\n"


STATIC_DIR = Path(__file__).resolve().parent / "static"


@app.get("/", include_in_schema=False)
def home():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
def health():
    return {"status": "ok"}


# plain def (not async) so slow llm calls run in a thread and don't block the server
def chat_response(message):
    """Used by both /chat and /voice."""
    try:
        result = app.state.assistant.handle(message)
        # memory is only saved once the whole stream is read
        answer = "".join(result["answer_stream"])
    except Exception as error:
        raise HTTPException(status_code=500, detail=f"Something went wrong: {error}")

    code = extract_python_code(answer) if result["route"] == "generate" else ""
    return {**answer_info(result), "answer": answer, "code": code}


@app.post("/chat")
def chat(request: ChatRequest):
    return chat_response(request.message)


@app.post("/chat/stream")
def chat_stream(request: ChatRequest):
    try:
        result = app.state.assistant.handle(request.message)
    except Exception as error:
        raise HTTPException(status_code=500, detail=f"Something went wrong: {error}")

    # one json object per line: info first, then the answer in pieces
    def lines():
        yield json_line({"type": "info", **answer_info(result)})
        try:
            for piece in result["answer_stream"]:
                yield json_line({"type": "text", "text": piece})
        except Exception as error:
            # the 200 status is already sent at this point, so the error goes inside the stream
            yield json_line({"type": "error", "detail": str(error)})
        yield json_line({"type": "done"})

    return StreamingResponse(lines(), media_type="application/x-ndjson")


@app.post("/run")
def run(request: RunRequest):
    return run_code(request.code)


@app.get("/memory")
def get_memory():
    assistant = app.state.assistant
    return {"memory": assistant.memory.data, "pending_request": assistant.pending_request}


@app.delete("/memory")
def clear_memory():
    assistant = app.state.assistant
    assistant.memory.clear()
    assistant.pending_request = None
    return {"status": "memory cleared"}


@app.post("/dataset")
def upload_dataset(file: UploadFile):
    try:
        info = load_dataset(file.file.read(), file.filename)
    except DatasetError as error:
        raise HTTPException(status_code=400, detail=str(error))
    return {**info, "description": describe_table()}


# lets you try the app without uploading anything
@app.post("/dataset/sample")
def use_sample_dataset():
    info = load_dataset(SAMPLE_DATASET_PATH.read_bytes(), SAMPLE_DATASET_PATH.name)
    return {**info, "description": describe_table()}


@app.get("/dataset")
def get_dataset():
    description = describe_table()
    if description is None:
        raise HTTPException(status_code=404, detail="No dataset loaded yet. Upload one with POST /dataset.")
    return {"description": description}


def table_description_or_404():
    description = describe_table()
    if description is None:
        raise HTTPException(status_code=404, detail="No dataset loaded yet. Upload one with POST /dataset.")
    return description


@app.post("/sql/generate")
def generate_sql_endpoint(request: QuestionRequest):
    description = table_description_or_404()
    try:
        return generate_sql(request.question, description, app.state.assistant.llm)
    except Exception as error:
        raise HTTPException(status_code=500, detail=f"Something went wrong: {error}")


@app.post("/sql/run")
def run_sql_endpoint(request: SqlRequest):
    table_description_or_404()
    try:
        return run_select(request.sql)
    except UnsafeSQLError as error:
        raise HTTPException(status_code=400, detail=f"Blocked: {error}")
    except sqlite3.Error as error:
        raise HTTPException(status_code=400, detail=f"SQL error: {error}")


@app.post("/analyze")
def analyze_endpoint(request: QuestionRequest):
    table_description_or_404()
    try:
        return analyze(request.question, app.state.assistant.llm)
    except Exception as error:
        raise HTTPException(status_code=500, detail=f"Something went wrong: {error}")


# auto detect sometimes thinks short arabic clips are english, so the user can pick
VoiceLanguage = Literal["auto", "ar", "en"]


def transcribe_upload(audio, language):
    # check the size without reading the whole file into memory
    audio.file.seek(0, 2)
    size_mb = audio.file.tell() / 1_000_000
    audio.file.seek(0)
    if size_mb > MAX_AUDIO_MB:
        raise HTTPException(status_code=400, detail=f"Audio is {size_mb:.1f} MB; the limit is {MAX_AUDIO_MB} MB.")

    try:
        result = transcribe(app.state.whisper, audio.file, language=None if language == "auto" else language)
    except Exception as error:
        raise HTTPException(status_code=400, detail=f"Could not read the audio file: {error}")

    if not result["text"]:
        raise HTTPException(status_code=400, detail="I couldn't hear any speech in the recording.")
    return result


@app.post("/transcribe")
def transcribe_endpoint(audio: UploadFile, language: VoiceLanguage = Form("auto")):
    return transcribe_upload(audio, language)


@app.post("/voice")
def voice(audio: UploadFile, language: VoiceLanguage = Form("auto")):
    transcript = transcribe_upload(audio, language)
    response = chat_response(transcript["text"])
    # whisper step goes first so the trace shows the full pipeline
    response["trace"].insert(0, f"Faster-Whisper ({transcript['language']}, {transcript['seconds']}s) -> {transcript['text']}")
    return {"transcript": transcript, **response}
