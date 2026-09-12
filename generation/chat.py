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

CHECKPOINT_PATH = PROJECT_ROOT / "checkpoints" / "geniee_sft_best.pt"
INSTRUCTION_SPLIT_DIR = PROJECT_ROOT / "data" / "processed" / "splits"


def _question_terms(text):
    stop_words = {
        "a", "an", "and", "are", "can", "do", "does", "how", "in",
        "is", "it", "of", "on", "or", "the", "to", "what", "when",
        "why", "with",
    }
    return {
        word
        for word in re.findall(r"[a-z0-9]+", text.lower())
        if word not in stop_words
    }


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


class GenieeChat:
    """Interactive Geniee chat using the same format used during SFT."""

    def __init__(self, generator, system_prompt=DEFAULT_SYSTEM_PROMPT, knowledge_base=None):
        self.generator = generator
        self.conversation = GenieeConversation(system_prompt=system_prompt, max_turns=6)
        self.knowledge_base = knowledge_base if knowledge_base is not None else _load_instruction_answers()

    def _retrieve_answer(self, user_text):
        query = user_text.strip().lower()
        query_terms = _question_terms(query)
        if not query_terms:
            return None

        exact_match = next(
            (answer for question, answer, _ in self.knowledge_base if question.lower() == query),
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
            if overlap and score > best_score:
                best_score = score
                best_answer = answer

        return best_answer if best_score >= 0.5 else None

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

    def respond(self, user_text, max_new_tokens=80):
        user_text = user_text.strip()
        if not user_text:
            return ""

        if max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be greater than 0")

        self.conversation.add_user(user_text)
        retrieved = self._retrieve_answer(user_text)
        if retrieved:
            self.conversation.add_assistant(retrieved)
            return retrieved

        prompt = self._fit_prompt_to_context(max_new_tokens)

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
                print("\nGeniee: " + self.respond(user_input))
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
