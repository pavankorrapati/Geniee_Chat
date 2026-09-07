# from pathlib import Path

# from generation.generate import load_geniee


# def get_checkpoint_path():

#     project_root = (
#         Path(__file__).resolve().parent.parent
#     )

#     checkpoint_path = (
#         project_root
#         / "checkpoints"
#         / "geniee_epoch_5.pt"
#     )

#     return checkpoint_path


# def test_load_geniee():

#     checkpoint_path = (
#         get_checkpoint_path()
#     )

#     if not checkpoint_path.exists():

#         raise FileNotFoundError(
#             "Expected trained checkpoint was not found:\n"
#             f"{checkpoint_path}\n\n"
#             "Run:\n"
#             "python training/train.py"
#         )

#     generator = load_geniee(
#         checkpoint_path
#     )

#     assert generator is not None

#     assert generator.model is not None

#     assert generator.tokenizer is not None


# def test_generate_text():

#     checkpoint_path = (
#         get_checkpoint_path()
#     )

#     if not checkpoint_path.exists():

#         raise FileNotFoundError(
#             "Expected trained checkpoint was not found:\n"
#             f"{checkpoint_path}"
#         )

#     generator = load_geniee(
#         checkpoint_path
#     )

#     prompt = "Geniee is"

#     generated_text = (
#         generator.generate(
#             prompt,
#             max_new_tokens=10
#         )
#     )

#     print()
#     print(
#         "Prompt:",
#         prompt
#     )

#     print(
#         "Generated:",
#         generated_text
#     )

#     assert isinstance(
#         generated_text,
#         str
#     )

#     assert len(
#         generated_text
#     ) > 0


from pathlib import Path

import torch

from generation.generate import load_geniee


def get_checkpoint_path():

    project_root = (
        Path(__file__).resolve().parent.parent
    )

    return (
        project_root
        / "checkpoints"
        / "geniee_epoch_5.pt"
    )


def get_generator():

    checkpoint_path = (
        get_checkpoint_path()
    )

    if not checkpoint_path.exists():

        raise FileNotFoundError(
            "Expected trained checkpoint was not found:\n"
            f"{checkpoint_path}\n\n"
            "Run:\n"
            "python training/train.py"
        )

    return load_geniee(
        checkpoint_path
    )


# ==========================================================
# Model loading
# ==========================================================

def test_load_geniee():

    generator = get_generator()

    assert generator is not None

    assert generator.model is not None

    assert generator.tokenizer is not None


# ==========================================================
# Basic generation
# ==========================================================

def test_generate_text():

    generator = get_generator()

    prompt = "Geniee is"

    generated_text = (
        generator.generate(
            prompt,
            max_new_tokens=10
        )
    )

    print()
    print(
        "Prompt:",
        prompt
    )

    print(
        "Generated:",
        generated_text
    )

    assert isinstance(
        generated_text,
        str
    )

    assert len(
        generated_text
    ) > 0


# ==========================================================
# Greedy generation
# ==========================================================

def test_greedy_generation():

    generator = get_generator()

    text = generator.generate(
        "Geniee is",
        max_new_tokens=10,
        do_sample=False
    )

    print()
    print(
        "Greedy output:",
        text
    )

    assert isinstance(
        text,
        str
    )

    assert len(text) > 0


# ==========================================================
# Sampling generation
# ==========================================================

def test_sampling_generation():

    generator = get_generator()

    text = generator.generate(
        "Geniee is",
        max_new_tokens=10,
        temperature=0.8,
        top_k=20,
        top_p=0.9,
        repetition_penalty=1.1,
        do_sample=True
    )

    print()
    print(
        "Sampled output:",
        text
    )

    assert isinstance(
        text,
        str
    )

    assert len(text) > 0


# ==========================================================
# Token generation
# ==========================================================

def test_generate_tokens():

    generator = get_generator()

    token_ids = generator.tokenizer.encode(
        "Geniee is",
        add_bos=True,
        add_eos=False
    )

    input_ids = torch.tensor(
        [token_ids],
        dtype=torch.long
    )

    output_ids = (
        generator.generate_tokens(
            input_ids,
            max_new_tokens=5,
            do_sample=False
        )
    )

    print()
    print(
        "Input shape:",
        input_ids.shape
    )

    print(
        "Output shape:",
        output_ids.shape
    )

    assert output_ids.ndim == 2

    assert (
        output_ids.shape[0]
        == 1
    )

    assert (
        output_ids.shape[1]
        >= input_ids.shape[1]
    )


# ==========================================================
# Temperature validation
# ==========================================================

def test_invalid_temperature():

    generator = get_generator()

    try:

        generator.generate(
            "Geniee",
            max_new_tokens=2,
            temperature=0
        )

        assert False

    except ValueError as error:

        assert (
            "temperature"
            in str(error)
        )


# ==========================================================
# Top-p validation
# ==========================================================

def test_invalid_top_p():

    generator = get_generator()

    try:

        generator.generate(
            "Geniee",
            max_new_tokens=2,
            top_p=1.5
        )

        assert False

    except ValueError as error:

        assert (
            "top_p"
            in str(error)
        )