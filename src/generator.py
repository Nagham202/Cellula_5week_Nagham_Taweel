from langchain_core.prompts import ChatPromptTemplate

from src.prompts import GENERATOR_HUMAN, GENERATOR_SYSTEM
from src.retriever import format_context

GENERATOR_PROMPT = ChatPromptTemplate.from_messages([
    ("system", GENERATOR_SYSTEM),
    ("human", GENERATOR_HUMAN),
])


def stream_generation(request, documents, llm, memory_text="Conversation memory: no previous conversation."):
    messages = GENERATOR_PROMPT.format_messages(request=request, context=format_context(documents),
                                                memory=memory_text)
    for piece in llm.stream(messages):
        if piece.content:
            yield piece.content
