from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import main
from src.memory import ConversationMemory

AUDIO_DIR = Path(__file__).resolve().parent.parent / "data" / "audio_tests"


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    with TestClient(main.app) as client:
        # temp memory so the tests don't touch my real one
        main.app.state.assistant.memory = ConversationMemory(tmp_path_factory.mktemp("memory") / "memory.json")
        client.post("/dataset/sample")
        yield client
        client.post("/dataset/sample")


def test_home_page(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "<html" in response.text


def test_health(client):
    assert client.get("/health").json() == {"status": "ok"}


def test_sample_dataset(client):
    body = client.post("/dataset/sample").json()
    assert body["table"] == "sample_sales"
    assert body["rows"] == 500
    assert "Ramallah" in body["description"]


def test_upload_cleans_column_names(client):
    csv = "Unit Price ($),City,Order\n10,Ramallah,1\n20,Nablus,2\n".encode()
    body = client.post("/dataset", files={"file": ("My Sales.csv", csv, "text/csv")}).json()
    assert body["table"] == "my_sales"
    assert list(body["columns"]) == ["unit_price", "city", "order_value"]
    assert body["rows"] == 2
    client.post("/dataset/sample")


def test_upload_wrong_file_type(client):
    response = client.post("/dataset", files={"file": ("notes.txt", b"hello", "text/plain")})
    assert response.status_code == 400


def test_sql_run_select(client):
    body = client.post("/sql/run", json={"sql": "SELECT COUNT(*) FROM sample_sales"}).json()
    assert body["rows"] == [[500]]


@pytest.mark.parametrize("sql", [
    "DELETE FROM sample_sales",
    "DROP TABLE sample_sales",
    "SELECT 1; DROP TABLE sample_sales",
    "WITH x AS (SELECT 1) DELETE FROM sample_sales",
])
def test_sql_run_blocks_writes(client, sql):
    response = client.post("/sql/run", json={"sql": sql})
    assert response.status_code == 400
    assert response.json()["detail"].startswith("Blocked")


def test_sql_run_bad_column(client):
    response = client.post("/sql/run", json={"sql": "SELECT nope FROM sample_sales"})
    assert response.status_code == 400
    assert "no such column" in response.json()["detail"]


def test_data_unchanged_after_attacks(client):
    body = client.post("/sql/run", json={"sql": "SELECT COUNT(*), SUM(total) FROM sample_sales"}).json()
    assert body["rows"][0][0] == 500


def test_run_code(client):
    body = client.post("/run", json={"code": "print(sum(range(10)))"}).json()
    assert body["stdout"].strip() == "45"
    assert body["return_code"] == 0


def test_chat_rejects_empty_message(client):
    assert client.post("/chat", json={"message": ""}).status_code == 422


def test_memory_clear(client):
    assert client.delete("/memory").json() == {"status": "memory cleared"}
    assert client.get("/memory").json()["memory"]["turns"] == []


def test_transcribe_rejects_non_audio(client):
    response = client.post("/transcribe", files={"audio": ("notes.txt", b"hello", "text/plain")})
    assert response.status_code == 400


def test_voice_rejects_unknown_language(client):
    response = client.post("/voice", files={"audio": ("x.m4a", b"", "audio/mp4")}, data={"language": "fr"})
    assert response.status_code == 422


@pytest.mark.skipif(not (AUDIO_DIR / "en_2.m4a").exists(), reason="my voice recordings aren't in the repo")
def test_transcribe_english_clip(client):
    with open(AUDIO_DIR / "en_2.m4a", "rb") as audio:
        body = client.post("/transcribe", files={"audio": ("en_2.m4a", audio, "audio/mp4")}).json()
    assert body["language"] == "en"
    assert "product" in body["text"].lower()


@pytest.mark.skipif(not (AUDIO_DIR / "ar_1.m4a").exists(), reason="my voice recordings aren't in the repo")
def test_transcribe_arabic_with_language_picker(client):
    with open(AUDIO_DIR / "ar_1.m4a", "rb") as audio:
        body = client.post("/transcribe", files={"audio": ("ar_1.m4a", audio, "audio/mp4")},
                           data={"language": "ar"}).json()
    assert body["language"] == "ar"
    assert "المبيعات" in body["text"]
