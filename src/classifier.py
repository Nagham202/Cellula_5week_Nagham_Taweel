from langchain_core.prompts import ChatPromptTemplate

from src.dataset import dataset_summary
from src.llm import get_llm
from src.prompts import CLASSIFIER_HUMAN, CLASSIFIER_SYSTEM

EXPLAIN = "Code Explanation"
GENERATE = "Code Generation"
DATA = "Data Analysis"

CLASSIFIER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", CLASSIFIER_SYSTEM),
    ("human", CLASSIFIER_HUMAN),
])


def parse_label(raw_answer):
    text = raw_answer.strip().lower()
    # only look for the key word, the model sometimes adds extra text
    if "explanation" in text:
        return EXPLAIN
    if "generation" in text:
        return GENERATE
    if "analysis" in text:
        return DATA
    return None


def classify_intent(message, llm, memory_text="Conversation memory: no previous conversation.", dataset_text=None):
    # the table's column names help it spot questions about the data
    if dataset_text is None:
        dataset_text = dataset_summary()
    messages = CLASSIFIER_PROMPT.format_messages(message=message, memory=memory_text, dataset=dataset_text)
    raw_answer = llm.invoke(messages).content
    label = parse_label(raw_answer)

    if label is None:
        print(f"[classifier] could not parse {raw_answer!r}, using {GENERATE}")
        # generation is the safer guess because it checks the knowledge base first
        label = GENERATE
    return label, raw_answer


if __name__ == "__main__":
    llm = get_llm()
    examples = [
        "Explain this code:\nnums = [x * x for x in range(5)]",
        "Write a function that checks if a number is prime",
        "قديش بعنا بنابلس سنة 2024؟",
    ]
    for example in examples:
        label, raw_answer = classify_intent(example, llm)
        print(f"{label:17s} (raw: {raw_answer!r})  <- {example.splitlines()[0]}")
