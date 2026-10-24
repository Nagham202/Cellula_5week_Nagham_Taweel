import json
import re
from pathlib import Path

from src.config import AUDIO_TESTS_DIR
from src.speech_to_text import load_whisper_model, transcribe

EVAL_DIR = Path(__file__).resolve().parent
RESULTS_PATH = EVAL_DIR / "results_speech_to_text.json"

# usual arabic normalization, so different alef forms don't count as errors
ARABIC_LETTER_FORMS = str.maketrans({"أ": "ا", "إ": "ا", "آ": "ا", "ة": "ه", "ى": "ي"})


def words(text):
    text = text.lower().translate(ARABIC_LETTER_FORMS)
    return re.sub(r"[^\w\s]", " ", text).split()


def edit_distance(expected, heard):
    previous = list(range(len(heard) + 1))
    for i, expected_item in enumerate(expected, start=1):
        current = [i]
        for j, heard_item in enumerate(heard, start=1):
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (expected_item != heard_item)))
        previous = current
    return previous[-1]


def error_rates(expected_text, heard_text):
    expected_words, heard_words = words(expected_text), words(heard_text)
    expected_chars, heard_chars = " ".join(expected_words), " ".join(heard_words)
    wer = edit_distance(expected_words, heard_words) / max(len(expected_words), 1)
    cer = edit_distance(expected_chars, heard_chars) / max(len(expected_chars), 1)
    return round(wer, 2), round(cer, 2)


if __name__ == "__main__":
    expected = json.loads((AUDIO_TESTS_DIR / "expected.json").read_text(encoding="utf-8"))
    model = load_whisper_model()
    results = []

    for file_name, said in expected.items():
        path = AUDIO_TESTS_DIR / file_name
        if not path.exists():
            continue
        language = file_name[:2]
        for mode in ("auto", language):
            result = transcribe(model, str(path), language=None if mode == "auto" else mode)
            wer, cer = error_rates(said, result["text"])
            print(f"{file_name:13s} {mode:4s} -> [{result['language']}] WER {wer:4.2f}  CER {cer:4.2f}  "
                  f"{result['seconds']:4.1f}s  {result['text']}")
            results.append({"file": file_name, "language": language, "mode": mode, "said": said,
                            "heard": result["text"], "detected": result["language"],
                            "wer": wer, "cer": cer, "seconds": result["seconds"]})

    print()
    for language in ("en", "ar"):
        for mode in ("auto", "fixed"):
            subset = [r for r in results if r["language"] == language and (r["mode"] == "auto") == (mode == "auto")]
            if subset:
                average_wer = sum(r["wer"] for r in subset) / len(subset)
                average_cer = sum(r["cer"] for r in subset) / len(subset)
                wrong_language = sum(r["detected"] != language for r in subset)
                print(f"{language} {mode:5s}: WER {average_wer:.2f}  CER {average_cer:.2f}  "
                      f"wrong language {wrong_language}/{len(subset)}")

    RESULTS_PATH.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved to {RESULTS_PATH.name}")
