from pathlib import Path
import sys
PROJECT_ROOT = (
    Path(__file__).resolve().parent.parent
)

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from tokenizer.tokenizer import GenieeTokenizer


PROJECT_ROOT = Path(__file__).resolve().parent.parent

TOKENIZER_PATH = (
    PROJECT_ROOT
    / "tokenizer"
    / "artifacts"
    / "geniee.model"
)


def main():

    tokenizer = GenieeTokenizer(
        TOKENIZER_PATH
    )

    print("=" * 70)
    print("GENIEE TOKENIZER TEST")
    print("=" * 70)

    print(
        f"Vocabulary : {tokenizer.vocab_size}"
    )

    special_tokens = [
        "<|system|>",
        "<|user|>",
        "<|assistant|>",
    ]

    print("\nSPECIAL TOKENS")
    print("-" * 70)

    for token in special_tokens:

        token_ids = tokenizer.encode(
            token,
            add_bos=False,
            add_eos=False
        )

        print(
            f"{token:15} -> {token_ids}"
        )

    print("\nCHAT PROMPT")
    print("-" * 70)

    prompt = (
        "<|system|>\n"
        "You are Geniee, an AI assistant specialized "
        "in software testing and test automation.\n"
        "<|user|>\n"
        "What is Selenium?\n"
        "<|assistant|>\n"
    )

    ids = tokenizer.encode(
        prompt,
        add_bos=True,
        add_eos=False
    )

    print("Token IDs:")
    print(ids)

    print("\nToken count:")
    print(len(ids))

    print("\nDecoded:")
    print(tokenizer.decode(ids))

    print("=" * 70)


if __name__ == "__main__":
    main()