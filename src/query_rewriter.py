from langchain_core.prompts import ChatPromptTemplate

from src.config import MAX_FOLLOW_UP_CHARS
from src.prompts import DATA_STANDALONE_HUMAN, DATA_STANDALONE_SYSTEM, STANDALONE_HUMAN, STANDALONE_SYSTEM

STANDALONE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", STANDALONE_SYSTEM),
    ("human", STANDALONE_HUMAN),
])

DATA_STANDALONE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", DATA_STANDALONE_SYSTEM),
    ("human", DATA_STANDALONE_HUMAN),
])


def make_standalone_request(message, memory, llm):
    # nothing to fill in on the first message, and long messages are already complete
    if not memory.data["turns"] or len(message) > MAX_FOLLOW_UP_CHARS:
        return message

    messages = STANDALONE_PROMPT.format_messages(
        message=message,
        memory=memory.as_prompt_text(include_code=False),
    )
    rewritten = llm.invoke(messages).content.strip()
    return rewritten or message


def make_standalone_question(message, memory, llm):
    turns = memory.data["turns"]
    # only rewrite after a data answer, saves a groq call on every other question
    if not turns or turns[-1]["route"] != "data_analysis" or len(message) > MAX_FOLLOW_UP_CHARS:
        return message

    messages = DATA_STANDALONE_PROMPT.format_messages(
        message=message,
        memory=memory.as_prompt_text(include_code=False),
    )
    rewritten = llm.invoke(messages).content.strip()
    return rewritten or message
