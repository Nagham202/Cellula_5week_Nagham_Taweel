import time

from src.classifier import DATA, EXPLAIN, classify_intent
from src.code_runner import extract_python_code, run_code
from src.data_analyst import analyze
from src.explainer import stream_explanation
from src.feedback import is_solution, save_solution
from src.generator import stream_generation
from src.knowledge_base import load_vector_store
from src.llm import get_llm
from src.memory import ConversationMemory
from src.query_rewriter import make_standalone_question, make_standalone_request
from src.relevance_judge import RELEVANT, judge_relevance
from src.result_explainer import answer_language, stream_result_explanation
from src.retriever import retrieve

NOT_RELEVANT_MESSAGE = (
    "I couldn't find relevant knowledge for your request. "
    "Could you provide the correct solution or reference so I can learn it for future interactions?"
)


# wraps a fixed reply so it looks like a stream
def single_message(text):
    yield text


class CodingAssistant:
    def __init__(self):
        self.llm = get_llm()
        self.vector_store = load_vector_store()
        self.pending_request = None
        self.memory = ConversationMemory()

    def handle(self, message):
        result = self.route(message)
        # the full answer only exists after streaming ends, so memory is saved then
        result["answer_stream"] = self.remember_when_done(message, result["route"], result["answer_stream"])
        return result

    def remember_when_done(self, message, route, answer_stream):
        full_answer = ""
        for piece in answer_stream:
            full_answer += piece
            yield piece
        self.memory.add_turn(message, route, full_answer)

    def route(self, message):
        trace = []

        # if I just asked the user to teach me, check if this message is the solution
        if self.pending_request:
            learned = self.feedback_path(message, trace)
            if learned:
                return learned

        short_memory = self.memory.as_prompt_text(include_code=False)
        intent, raw_answer = classify_intent(message, self.llm, short_memory)
        trace.append(f"Classifier -> {intent}")

        if intent == EXPLAIN:
            return self.explain_path(message, intent, trace)
        if intent == DATA:
            return self.data_path(message, intent, trace)
        return self.generate_path(message, intent, trace)

    def feedback_path(self, message, trace):
        request = self.pending_request
        looks_like_solution, raw_answer = is_solution(request, message, self.llm)

        self.pending_request = None

        if not looks_like_solution:
            trace.append("Feedback check -> New Request (not saved)")
            return None

        trace.append("Feedback check -> Solution")
        doc_id, chunk_count = save_solution(request, message, self.vector_store)
        trace.append(f"Embedded + metadata + inserted into Chroma as {doc_id} ({chunk_count} chunk)")

        reply = (f"Thank you! I saved your solution to my knowledge base as `{doc_id}`.\n\n"
                 f"Next time someone asks for something like \"{request}\", I can use it.")
        return {"intent": "Feedback", "route": "learn", "trace": trace,
                "sources": [], "answer_stream": single_message(reply)}

    def explain_path(self, message, intent, trace):
        trace.append("Explain path: LLM only (no retrieval)")
        answer_stream = stream_explanation(message, self.llm, self.memory.as_prompt_text())
        return {"intent": intent, "route": "explain", "trace": trace,
                "sources": [], "answer_stream": answer_stream}

    def data_path(self, message, intent, trace):
        # turns follow-ups like "now only for 2024" into a full question
        question = make_standalone_question(message, self.memory, self.llm)
        if question != message:
            trace.append(f"Follow-up rewritten as: {question}")

        # the explanation is streamed below instead
        result = analyze(question, self.llm, explain=False)
        trace.extend(result["trace"])
        data = {key: value for key, value in result.items() if key not in ("trace", "explanation")}

        if result["error"]:
            if answer_language(question) == "Arabic":
                reply = f"ما قدرت أجاوب على هذا السؤال من البيانات.\n\n{result['error']}"
            else:
                reply = f"I couldn't answer this from the data.\n\n{result['error']}"
            answer_stream = single_message(reply)
        else:
            trace.append(f"LLM explains the result in {answer_language(question)}")
            answer_stream = stream_result_explanation(result["question"], result, self.llm)

        return {"intent": intent, "route": "data_analysis", "trace": trace,
                "sources": [], "data": data, "answer_stream": answer_stream}

    def generate_path(self, message, intent, trace):
        request = make_standalone_request(message, self.memory, self.llm)
        if request != message:
            trace.append(f"Follow-up rewritten as: {request}")

        results = retrieve(request, self.vector_store)
        documents = [document for document, distance in results]
        found = ", ".join(f"{doc.metadata['doc_id']} ({distance:.2f})" for doc, distance in results)
        trace.append(f"Retrieved top-{len(results)}: {found}")

        memory_text = self.memory.as_prompt_text()
        # chroma always returns something, so check it's actually useful
        verdict, raw_answer = judge_relevance(request, documents, self.llm, memory_text)
        trace.append(f"Relevance judge -> {verdict}")

        if verdict == RELEVANT:
            trace.append("LLM code generation with the retrieved context")
            answer_stream = stream_generation(message, documents, self.llm, memory_text)
            return {"intent": intent, "route": "generate", "trace": trace,
                    "sources": results, "answer_stream": answer_stream}

        # remember it so the next message can be saved as the solution
        self.pending_request = request
        trace.append("Not relevant -> asking the user for a solution")
        return {"intent": intent, "route": "not_relevant", "trace": trace,
                "sources": results, "answer_stream": single_message(NOT_RELEVANT_MESSAGE)}


if __name__ == "__main__":
    assistant = CodingAssistant()
    assistant.memory.clear()

    conversation = [
        "What are the total sales per city?",
        "now only for 2024",
        "كم طلبية بنابلس سنة ٢٠٢٣؟",
        "In Python, write a function that returns the sum of the even numbers in a list",
    ]

    for message in conversation:
        print("=" * 70)
        print("USER:", message.splitlines()[0])
        start = time.time()
        result = assistant.handle(message)
        answer = "".join(result["answer_stream"])
        print("TRACE:\n  " + "\n  ".join(result["trace"]))
        if result.get("data"):
            print("ROWS:", result["data"]["rows"][:6])
        print(f"ASSISTANT ({result['route']}, {time.time() - start:.1f}s):")
        print(answer[:700] + ("\n..." if len(answer) > 700 else ""))

    print("=" * 70)
    print("SAVED MEMORY:", len(assistant.memory.data["turns"]), "turns,",
          len(assistant.memory.data["generated_code"]), "generated code,",
          "preferences =", assistant.memory.data["preferences"])
