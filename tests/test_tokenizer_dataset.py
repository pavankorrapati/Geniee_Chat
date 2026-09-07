from pathlib import Path

import torch

from tokenizer.tokenizer import GenieeTokenizer
from data.dataset import GenieeTextDataset


def test_real_text_to_dataset():

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
    # Real text
    # --------------------------------------------------

    text = """
    Geniee is an artificial intelligence
    language model.
    Geniee learns from text.
    """

    # --------------------------------------------------
    # Encode
    # --------------------------------------------------

    token_ids = tokenizer.encode(
        text,
        add_bos=True,
        add_eos=True
    )

    print()
    print("Token IDs:")
    print(token_ids)

    # --------------------------------------------------
    # Convert to tensor
    # --------------------------------------------------

    token_tensor = torch.tensor(
        token_ids,
        dtype=torch.long
    )

    # --------------------------------------------------
    # Dataset
    # --------------------------------------------------

    dataset = GenieeTextDataset(
        token_ids=token_tensor,
        max_seq_len=8,
        stride=4
    )

    # --------------------------------------------------
    # Validate
    # --------------------------------------------------

    assert len(dataset) > 0

    input_ids, target_ids = dataset[0]

    print()
    print("Input IDs:")
    print(input_ids)

    print()
    print("Target IDs:")
    print(target_ids)

    assert input_ids.shape == (
        8,
    )

    assert target_ids.shape == (
        8,
    )

    assert input_ids.dtype == torch.long

    assert target_ids.dtype == torch.long