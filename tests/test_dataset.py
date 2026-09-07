import torch

from data.dataset import GenieeTextDataset


def test_geniee_dataset():

    # --------------------------------------------------
    # Token sequence
    # --------------------------------------------------

    token_ids = torch.tensor(
        [10, 20, 30, 40, 50, 60, 70, 80],
        dtype=torch.long
    )

    max_seq_len = 4

    # --------------------------------------------------
    # Create dataset
    # --------------------------------------------------

    dataset = GenieeTextDataset(
        token_ids=token_ids,
        max_seq_len=max_seq_len,
        stride=1
    )

    # --------------------------------------------------
    # Number of samples
    #
    # 8 tokens
    # sequence length = 4
    # stride = 1
    #
    # Number = 4
    # --------------------------------------------------

    assert len(dataset) == 4

    # --------------------------------------------------
    # First sample
    # --------------------------------------------------

    input_ids, target_ids = dataset[0]

    print()
    print("Input :", input_ids)
    print("Target:", target_ids)

    assert torch.equal(
        input_ids,
        torch.tensor([10, 20, 30, 40])
    )

    assert torch.equal(
        target_ids,
        torch.tensor([20, 30, 40, 50])
    )

    # --------------------------------------------------
    # Second sample
    # --------------------------------------------------

    input_ids, target_ids = dataset[1]

    assert torch.equal(
        input_ids,
        torch.tensor([20, 30, 40, 50])
    )

    assert torch.equal(
        target_ids,
        torch.tensor([30, 40, 50, 60])
    )


def test_dataset_returns_long_tensors():

    token_ids = [
        1,
        2,
        3,
        4,
        5,
        6
    ]

    dataset = GenieeTextDataset(
        token_ids=token_ids,
        max_seq_len=4
    )

    input_ids, target_ids = dataset[0]

    assert input_ids.dtype == torch.long
    assert target_ids.dtype == torch.long


def test_dataset_length():

    token_ids = torch.arange(
        100,
        dtype=torch.long
    )

    dataset = GenieeTextDataset(
        token_ids=token_ids,
        max_seq_len=10,
        stride=5
    )

    expected_length = (
        (100 - 10 - 1) // 5
    ) + 1

    assert len(dataset) == expected_length