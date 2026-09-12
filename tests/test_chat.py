from generation.chat import GenieeChat


class FakeGenerator:
    class Model:
        class Config:
            max_seq_len = 256

        config = Config()

    class Tokenizer:
        def encode(self, text, **_kwargs):
            return text.split()

    model = Model()
    tokenizer = Tokenizer()

    def __init__(self):
        self.calls = 0

    def generate(self, *_args, **_kwargs):
        self.calls += 1
        return "model fallback"


def test_chat_uses_exact_instruction_answer():
    generator = FakeGenerator()
    chat = GenieeChat(
        generator,
        knowledge_base=[
            (
                "What is Python?",
                "Python is a programming language.",
                {"python"},
            )
        ],
    )

    assert chat.respond("What is Python?") == "Python is a programming language."
    assert generator.calls == 0


def test_chat_uses_model_for_unknown_question():
    generator = FakeGenerator()
    chat = GenieeChat(generator, knowledge_base=[])

    assert chat.respond("Explain something new") == "model fallback"
    assert generator.calls == 1
