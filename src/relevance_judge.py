from langchain_core.prompts import ChatPromptTemplate

from src.prompts import RELEVANCE_HUMAN, RELEVANCE_SYSTEM
from src.retriever import format_context

RELEVANT = "Relevant"
NOT_RELEVANT = "Not Relevant"

RELEVANCE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", RELEVANCE_SYSTEM),
    ("human", RELEVANCE_HUMAN),
])


def parse_verdict(raw_answer):
    text = raw_answer.strip().lower()
    # check "not relevant" first because it also contains "relevant"
    if "not relevant" in text or "irrelevant" in text:
        return NOT_RELEVANT
    if "relevant" in text:
        return RELEVANT
    return None


def judge_relevance(request, documents, llm, memory_text="Conversation memory: no previous conversation."):
    messages = RELEVANCE_PROMPT.format_messages(request=request, context=format_context(documents),
                                                memory=memory_text)
    raw_answer = llm.invoke(messages).content
    verdict = parse_verdict(raw_answer)

    if verdict is None:
        print(f"[judge] could not parse {raw_answer!r}, using {NOT_RELEVANT}")
        # if unsure, don't guess
        verdict = NOT_RELEVANT
    return verdict, raw_answer
