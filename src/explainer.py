from langchain_core.prompts import ChatPromptTemplate

from src.prompts import EXPLAINER_HUMAN, EXPLAINER_SYSTEM


EXPLAINER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", EXPLAINER_SYSTEM),
    ("human", EXPLAINER_HUMAN),
])


def stream_explanation(message, llm, memory_text="Conversation memory: no previous conversation."):
    messages = EXPLAINER_PROMPT.format_messages(message=message, memory=memory_text)
    for piece in llm.stream(messages):
        if piece.content:
            yield piece.content
