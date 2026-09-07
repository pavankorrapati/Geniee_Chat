from pathlib import Path

from tokenizer.tokenizer import GenieeTokenizer


def test_geniee_tokenizer():

    # --------------------------------------------------
    # Locate tokenizer
    # --------------------------------------------------

    project_root = (
        Path(__file__).resolve().parent.parent
    )

    model_path = (
        project_root
        / "tokenizer"
        / "artifacts"
        / "geniee.model"
    )

    # --------------------------------------------------
    # Create tokenizer
    # --------------------------------------------------

    tokenizer = GenieeTokenizer(
        model_path
    )

    # --------------------------------------------------
    # Test text
    # --------------------------------------------------

    text = (
        "Geniee is learning "
        "language modeling."
    )

    # --------------------------------------------------
    # Encode
    # --------------------------------------------------

    token_ids = tokenizer.encode(
        text
    )

    print()
    print("Text:")
    print(text)

    print()
    print("Token IDs:")
    print(token_ids)

    # --------------------------------------------------
    # Validate encoding
    # --------------------------------------------------

    assert isinstance(
        token_ids,
        list
    )

    assert len(token_ids) > 0

    assert all(
        isinstance(token_id, int)
        for token_id in token_ids
    )

    # --------------------------------------------------
    # Validate vocabulary
    # --------------------------------------------------

    print()
    print(
        "Vocabulary size:",
        tokenizer.vocab_size
    )

    assert tokenizer.vocab_size > 0

    # --------------------------------------------------
    # Decode
    # --------------------------------------------------

    decoded_text = tokenizer.decode(
        token_ids
    )

    print()
    print("Decoded:")
    print(decoded_text)

    assert isinstance(
        decoded_text,
        str
    )

    assert len(decoded_text) > 0

def test_special_tokens():

    project_root = (
        Path(__file__).resolve().parent.parent
    )

    model_path = (
        project_root
        / "tokenizer"
        / "artifacts"
        / "geniee.model"
    )

    tokenizer = GenieeTokenizer(
        model_path
    )

    token_ids = tokenizer.encode(
        "Hello Geniee",
        add_bos=True,
        add_eos=True
    )
    print(f"\nSpecial Tokens: {token_ids}")

    assert token_ids[0] == tokenizer.bos_id

    assert token_ids[-1] == tokenizer.eos_id    