import time

from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama

from src.config import (
    GROQ_API_KEY,
    GROQ_MODEL,
    LLM_PROVIDER,
    LLM_TEMPERATURE,
    OLLAMA_MODEL,
)


def get_llm(temperature=LLM_TEMPERATURE):
    if LLM_PROVIDER == "groq":
        if not GROQ_API_KEY:
            raise ValueError("GROQ_API_KEY is missing. Put it in the .env file (see .env.example).")
        # retry and wait instead of crashing when we hit the rate limit
        return ChatGroq(model=GROQ_MODEL, temperature=temperature, api_key=GROQ_API_KEY, max_retries=6)

    if LLM_PROVIDER == "ollama":
        return ChatOllama(model=OLLAMA_MODEL, temperature=temperature)

    raise ValueError(f"Unknown LLM_PROVIDER: {LLM_PROVIDER}. Use 'groq' or 'ollama'.")


if __name__ == "__main__":
    llm = get_llm()
    print(f"Provider: {LLM_PROVIDER}\n")

    start = time.time()
    response = llm.invoke("Reply with exactly one word: Hello")
    print(f"invoke -> {response.content!r}  ({time.time() - start:.2f}s)\n")

    print("stream -> ", end="")
    for piece in llm.stream("Count from 1 to 10 separated by spaces."):
        print(piece.content, end="", flush=True)
    print()
