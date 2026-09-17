from generation.chat import GenieeChat
from rag.retriever import LocalRetriever
from python_debugger import diagnose_traceback


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
        return "Explain something new clearly and directly."


class FakeWebRetriever:
    def retrieve(self, query):
        return [("Orbital mechanics describes the motion of objects under gravity.", "https://example.test")]


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


def test_traceback_returns_reason_fix_and_verification():
    traceback = '''Traceback (most recent call last):
  File "app.py", line 4, in load_user
    return users[user_id]
KeyError: 'user_id'
'''
    answer = diagnose_traceback(traceback)

    assert "KeyError" in answer
    assert "Reason" in answer
    assert "Recommended fix" in answer
    assert "Verify" in answer


def test_chat_handles_spaced_exception_question():
    generator = FakeGenerator()
    chat = GenieeChat(generator, knowledge_base=[])

    answer = chat.respond("What is index error?")

    assert "IndexError" in answer
    assert "sequence index" in answer
    assert generator.calls == 0


def test_chat_uses_model_for_unknown_question():
    generator = FakeGenerator()
    chat = GenieeChat(generator, knowledge_base=[], retriever=LocalRetriever([]))

    assert chat.respond("Explain something new") == "Explain something new clearly and directly."
    assert generator.calls == 1


def test_chat_uses_web_evidence_for_unknown_question():
    generator = FakeGenerator()
    chat = GenieeChat(
        generator,
        knowledge_base=[],
        retriever=LocalRetriever([]),
        web_retriever=FakeWebRetriever(),
    )

    assert chat.respond("Explain orbital mechanics") == (
        "Orbital mechanics describes the motion of objects under gravity."
    )
