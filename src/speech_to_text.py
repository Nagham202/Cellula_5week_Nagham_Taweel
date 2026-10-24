import json
import sys
import time

from faster_whisper import WhisperModel

from src.config import (
    AUDIO_TESTS_DIR,
    WHISPER_BEAM_SIZE,
    WHISPER_COMPUTE_TYPE,
    WHISPER_CPU_THREADS,
    WHISPER_DEVICE,
    WHISPER_LANGUAGES,
    WHISPER_MODEL,
)

AUDIO_EXTENSIONS = {".m4a", ".mp3", ".wav", ".aiff", ".ogg", ".webm", ".flac"}


def load_whisper_model():
    # the first run downloads the model (~1.6 GB), after that it loads from disk
    return WhisperModel(WHISPER_MODEL, device=WHISPER_DEVICE, compute_type=WHISPER_COMPUTE_TYPE,
                        cpu_threads=WHISPER_CPU_THREADS)


def transcribe(model, audio, language=None):
    """audio can be a path or a file object, language is "ar", "en" or None for auto."""
    start = time.time()
    segments, info = model.transcribe(
        audio,
        language=language,
        beam_size=WHISPER_BEAM_SIZE,
        # cuts silence, whisper sometimes makes up words there
        vad_filter=True,
        # every question is separate
        condition_on_previous_text=False,
    )
    # segments is a generator, the actual transcribing happens here
    text = " ".join(segment.text.strip() for segment in segments).strip()

    # short arabic clips sometimes get detected as persian or urdu, so retry with the likelier of ar/en
    if language is None and info.language not in WHISPER_LANGUAGES and info.all_language_probs:
        probabilities = dict(info.all_language_probs)
        best = max(WHISPER_LANGUAGES, key=lambda code: probabilities.get(code, 0))
        if hasattr(audio, "seek"):
            audio.seek(0)
        result = transcribe(model, audio, language=best)
        result["note"] = f"auto-detect said {info.language!r}, forced {best!r}"
        return result

    return {
        "text": text,
        "language": info.language,
        "language_probability": round(info.language_probability, 2),
        "audio_seconds": round(info.duration, 2),
        "seconds": round(time.time() - start, 2),
    }


if __name__ == "__main__":
    folder = AUDIO_TESTS_DIR if len(sys.argv) < 2 else sys.argv[1]
    files = sorted(path for path in folder.iterdir() if path.suffix.lower() in AUDIO_EXTENSIONS)
    if not files:
        raise SystemExit(f"No audio files in {folder}. Record a few questions first.")

    expected_path = folder / "expected.json"
    expected = json.loads(expected_path.read_text(encoding="utf-8")) if expected_path.exists() else {}

    start = time.time()
    model = load_whisper_model()
    print(f"Loaded {WHISPER_MODEL} ({WHISPER_COMPUTE_TYPE}, {WHISPER_DEVICE}) in {time.time() - start:.1f}s\n")

    for path in files:
        result = transcribe(model, str(path))
        print(f"=== {path.name}  ({result['audio_seconds']}s of audio -> {result['seconds']}s to transcribe)")
        print(f"auto:     [{result['language']} {result['language_probability']}] {result['text']}  {result.get('note', '')}")
        # files named ar_ / en_ also get a run with the language fixed
        language = path.name[:2]
        if language in ("ar", "en"):
            fixed = transcribe(model, str(path), language=language)
            print(f"fixed {language}: {fixed['text']}")
        if path.name in expected:
            print(f"said:     {expected[path.name]}")
        print()
