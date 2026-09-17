from pathlib import Path
import sys
import json
import re

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from chat.conversation import GenieeConversation
from chat.prompt import DEFAULT_SYSTEM_PROMPT
from generation.generate import load_geniee
from python_debugger import diagnose_traceback
from rag.retriever import LocalRetriever, load_corpus_documents
from rag.web_retriever import WebRetriever

CHECKPOINT_PATH = PROJECT_ROOT / "checkpoints" / "geniee_sft_best.pt"
INSTRUCTION_SPLIT_DIR = PROJECT_ROOT / "data" / "processed" / "splits"
CORPUS_DIR = PROJECT_ROOT / "corpus"
WEB_CORPUS_DIR = PROJECT_ROOT / "data" / "web" / "cleaned"
UNKNOWN_ANSWER = (
    "I do not have enough trained information to answer that accurately yet. "
    "Please add a trusted example or document for this topic."
)


def _question_terms(text):
    stop_words = {
        "a", "an", "and", "are", "can", "do", "does", "how", "in",
        "is", "it", "of", "on", "or", "the", "to", "what", "when",
        "why", "with", "describe", "explain", "tell", "give",
    }
    normalized = _normalize_question(text)
    return {
        word
        for word in re.findall(r"[a-z0-9]+", normalized.lower())
        if word not in stop_words
    }


def _normalize_question(text):
    normalized = text.casefold()
    normalized = re.sub(r"\([^)]*\)", " ", normalized)
    normalized = normalized.replace("-", " ")
    normalized = re.sub(r"[^a-z0-9\s]", " ", normalized)
    normalized = re.sub(r"\bsqli\b", "sql injection", normalized)
    normalized = re.sub(r"\bxss\b", "cross site scripting", normalized)
    exception_aliases = {
        "why do i get a keyerror": "what is a keyerror",
        "why do i get keyerror": "what is a keyerror",
        "why do i get a nameerror": "what is a nameerror",
        "why do i get a typeerror": "what is a typeerror",
        "why do i get an indexerror": "what is an indexerror",
        "why do i get an attributeerror": "what is an attributeerror",
        "why do i get a valueerror": "what is a valueerror",
    }
    for source, target in exception_aliases.items():
        normalized = normalized.replace(source, target)
    normalized = re.sub(r"\bbruteforce\b|\bbrute force\b", "brute force", normalized)
    normalized = normalized.replace("cross-site", "cross site")
    normalized = normalized.replace("vulnerability scanning", "vulnerability scan")
    normalized = normalized.replace("penetration testing", "penetration test")
    for source, target in {
        "zjanasena": "janasena",
        "janasena": "jana sena",
        "cheif": "chief",
        "andhras": "andhra",
        "seleniumj": "selenium",
        "as per": "",
        "in andhra pradesh": "",
        "largest film budget": "largest telugu film budget",
    }.items():
        normalized = normalized.replace(source, target)
    normalized = re.sub(
        r"^what is brute force attack$|^what is brute force attack in security testing$",
        "what is a brute force attack in security testing",
        normalized,
    )
    normalized = re.sub(
        r"^what is sql injection and how do you test for it$",
        "what is sql injection testing",
        normalized,
    )
    normalized = re.sub(
        r"^what is cross site scripting cross site scripting and what are its main types$",
        "what are the main types of cross site scripting",
        normalized,
    )
    normalized = re.sub(
        r"^how do you approach testing api security$",
        "how do you approach testing api security",
        normalized,
    )
    for exception_name in (
        "modulenotfounderror", "importerror", "syntaxerror", "indentationerror",
        "taberror", "nameerror", "typeerror", "valueerror", "indexerror",
        "keyerror", "attributeerror", "unboundlocalerror", "zerodivisionerror",
        "filenotfounderror", "permissionerror",
    ):
        if exception_name in normalized and normalized != f"what is a {exception_name}":
            article = "an" if exception_name[0] in "aeiou" else "a"
            normalized = f"what is {article} {exception_name}"
            break
    return " ".join(normalized.split())


def _load_instruction_answers():
    answers = []
    for split_path in sorted(INSTRUCTION_SPLIT_DIR.glob("*.jsonl")):
        for line in split_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            messages = record.get("messages", [])
            user = next((m["content"] for m in messages if m.get("role") == "user"), None)
            assistant = next((m["content"] for m in messages if m.get("role") == "assistant"), None)
            if user and assistant:
                answers.append((user.strip(), assistant.strip(), _question_terms(user)))
    return answers


def _load_local_documents():
    documents = load_corpus_documents(CORPUS_DIR)
    if WEB_CORPUS_DIR.exists():
        documents.extend(
            (
                f"web/{name}",
                text,
            )
            for name, text in load_corpus_documents(WEB_CORPUS_DIR)
        )
    return documents


class GenieeChat:
    """Interactive Geniee chat using the same format used during SFT."""

    def __init__(
        self,
        generator,
        system_prompt=DEFAULT_SYSTEM_PROMPT,
        knowledge_base=None,
        retriever=None,
        web_retriever=None,
    ):
        self.generator = generator
        self.conversation = GenieeConversation(system_prompt=system_prompt, max_turns=6)
        self.knowledge_base = knowledge_base if knowledge_base is not None else _load_instruction_answers()
        self.retriever = retriever if retriever is not None else LocalRetriever(
            _load_local_documents()
        )
        self.web_retriever = web_retriever if web_retriever is not None else WebRetriever()
        self.last_source = "local"
        self.last_sources: list[str] = []

    def _retrieve_answer(self, user_text):
        query = _normalize_question(user_text.strip())
        query_terms = _question_terms(query)
        if not query_terms:
            return None

        exact_match = next(
            (answer for question, answer, _ in self.knowledge_base if _normalize_question(question) == query),
            None,
        )
        if exact_match:
            return exact_match

        best_answer = None
        best_score = 0.0
        for _, answer, question_terms in self.knowledge_base:
            if not question_terms:
                continue
            overlap = len(query_terms & question_terms)
            score = overlap / len(query_terms | question_terms)
            query_coverage = overlap / len(query_terms)
            if query_coverage >= 0.8 and score > best_score:
                best_score = score
                best_answer = answer

        return best_answer if best_score >= 0.65 else None

    def _fit_prompt_to_context(self, max_new_tokens):
        """Drop oldest turns until prompt + generation fit the context window."""
        tokenizer = self.generator.tokenizer
        max_context = self.generator.model.config.max_seq_len
        while True:
            prompt = self.conversation.prompt()
            token_count = len(tokenizer.encode(prompt, add_bos=True, add_eos=False))
            if token_count + max_new_tokens <= max_context:
                return prompt
            messages = self.conversation.memory.messages
            if len(messages) <= 2:
                return prompt
            # Remove the oldest complete user/assistant pair.
            del messages[:2]

    @staticmethod
    def clean_response(text):
        if not text:
            return "I am sorry, I could not generate a response."
        for marker in ("<|user|>", "<|system|>", "<|assistant|>"):
            if marker in text:
                text = text.split(marker, 1)[0]
        text = text.replace("<|endoftext|>", "").replace("⁇", "")
        return text.strip() or "I am sorry, I could not generate a response."

    @staticmethod
    def _looks_unreliable(text, question=""):
        words = text.split()
        if len(words) < 5 or len(set(words)) < max(3, len(words) // 3):
            return True
        question_terms = _question_terms(question)
        response_terms = _question_terms(text)
        return bool(question_terms) and not question_terms.intersection(response_terms)

    def respond(self, user_text, max_new_tokens=80):
        user_text = user_text.strip()
        if not user_text:
            return ""

        if max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be greater than 0")

        self.conversation.add_user(user_text)
        self.last_source = "local"
        self.last_sources = []

        traceback_answer = diagnose_traceback(user_text)
        if traceback_answer:
            self.conversation.add_assistant(traceback_answer)
            return traceback_answer

        retrieved = self._retrieve_answer(user_text)
        if retrieved:
            self.conversation.add_assistant(retrieved)
            return retrieved

        if not self.retriever.retrieve(user_text):
            try:
                web_evidence = self.web_retriever.retrieve(user_text)
            except (OSError, ValueError, KeyError, TypeError):
                web_evidence = []
            if web_evidence:
                response = " ".join(sentence for sentence, _ in web_evidence)
                self.last_source = "web"
                self.last_sources = [url for _, url in web_evidence]
                self.conversation.add_assistant(response)
                return response

        prompt = self._fit_prompt_to_context(max_new_tokens)
        references = self.retriever.retrieve(user_text)
        if references:
            reference_text = "\n\nRelevant local reference:\n" + "\n\n".join(
                f"[{name}] {text[:1200]}" for name, text in references
            )
            prompt_head, assistant_marker = prompt.rsplit("<|assistant|>\n", 1)
            prompt = prompt_head + reference_text + "\n<|assistant|>\n" + assistant_marker

        try:
            raw = self.generator.generate(
                prompt,
                max_new_tokens=max_new_tokens,
                temperature=0.7,
                top_k=30,
                top_p=0.90,
                repetition_penalty=1.05,
                do_sample=False,
                return_full_text=False,
            )
        except (OSError, RuntimeError, ValueError):
            self.conversation.memory.messages.pop()
            raise

        response = self.clean_response(raw)
        if references and self._looks_unreliable(response, user_text):
            evidence = self.retriever.best_sentences(user_text)
            if evidence:
                response = " ".join(evidence)
        elif not references and self._looks_unreliable(response, user_text):
            try:
                web_evidence = self.web_retriever.retrieve(user_text)
            except Exception:
                web_evidence = []
            if web_evidence:
                response = " ".join(sentence for sentence, _ in web_evidence)
                self.last_source = "web"
                self.last_sources = [url for _, url in web_evidence]
            else:
                response = UNKNOWN_ANSWER
        self.conversation.add_assistant(response)
        return response

    def start(self):
        print("\n" + "=" * 70)
        print("GENIEE CHAT")
        print("=" * 70)
        print("Commands: exit | quit | clear")
        print("Model: SFT instruction-tuned Geniee")

        while True:
            try:
                user_input = input("\nYou: ").strip()
            except (KeyboardInterrupt, EOFError):
                print("\nGoodbye!")
                break

            if not user_input:
                continue
            if user_input.lower() in {"exit", "quit"}:
                print("Goodbye!")
                break
            if user_input.lower() == "clear":
                self.conversation.clear()
                print("Conversation cleared.")
                continue

            try:
                response = self.respond(user_input)
                label = "Geniee (web-grounded)" if self.last_source == "web" else "Geniee"
                print(f"\n{label}: {response}")
            except Exception as exc:
                print(f"\nGeneration error: {exc}")


def load_chatbot():
    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"SFT checkpoint not found: {CHECKPOINT_PATH}\n"
            "Run training/train_pretrain.py first, then training/train_sft.py."
        )
    return GenieeChat(load_geniee(CHECKPOINT_PATH))


def main():
    load_chatbot().start()


if __name__ == "__main__":
    main()
