from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from chat.conversation import GenieeConversation
from chat.prompt import DEFAULT_SYSTEM_PROMPT
from generation.generate import load_geniee

CHECKPOINT_PATH = PROJECT_ROOT / "checkpoints" / "geniee_sft_best.pt"


class GenieeChat:
    """Interactive Geniee chat using the same format used during SFT."""

    def __init__(self, generator, system_prompt=DEFAULT_SYSTEM_PROMPT):
        self.generator = generator
        self.conversation = GenieeConversation(system_prompt=system_prompt, max_turns=6)

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
            return "I’m sorry, I could not generate a response."
        for marker in ("<|user|>", "<|system|>", "<|assistant|>"):
            if marker in text:
                text = text.split(marker, 1)[0]
        return text.strip() or "I’m sorry, I could not generate a response."

    def respond(self, user_text):
        user_text = user_text.strip()
        if not user_text:
            return ""

        self.conversation.add_user(user_text)
        max_new_tokens = 80
        prompt = self._fit_prompt_to_context(max_new_tokens)

        try:
            raw = self.generator.generate(
                prompt,
                max_new_tokens=max_new_tokens,
                temperature=0.65,
                top_k=30,
                top_p=0.90,
                repetition_penalty=1.05,
                do_sample=True,
                return_full_text=False,
            )
        except Exception:
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
