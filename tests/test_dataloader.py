import torch
from torch.utils.data import DataLoader

from data.dataset import GenieeTextDataset
def test_dataloader_input_target_alignment():

    # --------------------------------------------------
    # Simple predictable token IDs
    # --------------------------------------------------

    token_ids = torch.tensor(
        [
            10,
            20,
            30,
            40,
            50,
            60,
            70,
            80,
            90,
            100
        ],
        dtype=torch.long
    )

    # --------------------------------------------------
    # Create dataset
    # --------------------------------------------------

    dataset = GenieeTextDataset(
        token_ids=token_ids,
        max_seq_len=4,
        stride=1
    )

    # --------------------------------------------------
    # DataLoader
    # --------------------------------------------------

    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False
    )

    # --------------------------------------------------
    # Get first batch
    # --------------------------------------------------

    input_ids, target_ids = next(
        iter(dataloader)
    )

    # --------------------------------------------------
    # Remove batch dimension
    # --------------------------------------------------

    input_ids = input_ids[0]

    target_ids = target_ids[0]

    # --------------------------------------------------
    # Validate input
    # --------------------------------------------------

    expected_input_ids = torch.tensor(
        [10, 20, 30, 40],
        dtype=torch.long
    )

    assert torch.equal(
        input_ids,
        expected_input_ids
    )

    # --------------------------------------------------
    # Validate target
    # --------------------------------------------------

    expected_target_ids = torch.tensor(
        [20, 30, 40, 50],
        dtype=torch.long
    )

    assert torch.equal(
        target_ids,
        expected_target_ids
    )

    # --------------------------------------------------
    # Validate shape
    # --------------------------------------------------

    assert input_ids.shape == (
        4,
    )

    assert target_ids.shape == (
        4,
    )

# def test_geniee_dataloader():

#     # --------------------------------------------------
#     # Create token sequence
#     # --------------------------------------------------

#     token_ids = torch.arange(
#         100,
#         dtype=torch.long
#     )

#     # --------------------------------------------------
#     # Dataset configuration
#     # --------------------------------------------------

#     max_seq_len = 8
#     stride = 8
#     batch_size = 2

#     # --------------------------------------------------
#     # Create Geniee dataset
#     # --------------------------------------------------

#     dataset = GenieeTextDataset(
#         token_ids=token_ids,
#         max_seq_len=max_seq_len,
#         stride=stride
#     )

#     # --------------------------------------------------
#     # Create DataLoader
#     # --------------------------------------------------

#     dataloader = DataLoader(
#         dataset,
#         batch_size=batch_size,
#         shuffle=False
#     )

#     # --------------------------------------------------
#     # Get first batch
#     # --------------------------------------------------

#     input_ids, target_ids = next(
#         iter(dataloader)
#     )

#     # --------------------------------------------------
#     # Display information
#     # --------------------------------------------------

#     print()

#     print(
#         "Dataset length:",
#         len(dataset)
#     )

#     print(
#         "Input batch shape:",
#         input_ids.shape
#     )

#     print(
#         "Target batch shape:",
#         target_ids.shape
#     )

#     print(
#         "Input batch:"
#     )

#     print(input_ids)

#     print(
#         "Target batch:"
#     )

#     print(target_ids)

#     # --------------------------------------------------
#     # Expected shapes
#     #
#     # batch_size = 2
#     # max_seq_len = 8
#     #
#     # Therefore:
#     #
#     # input  -> [2, 8]
#     # target -> [2, 8]
#     # --------------------------------------------------

#     assert input_ids.shape == (
#         batch_size,
#         max_seq_len
#     )

#     assert target_ids.shape == (
#         batch_size,
#         max_seq_len
#     )

#     # --------------------------------------------------
#     # Verify tensor type
#     # --------------------------------------------------

#     assert input_ids.dtype == torch.long

#     assert target_ids.dtype == torch.long


# def test_dataloader_input_target_alignment():

#     # --------------------------------------------------
#     # Simple predictable token IDs
#     # --------------------------------------------------

#     token_ids = torch.tensor(
#         [
#             10,
#             20,
#             30,
#             40,
#             50,
#             60,
#             70,
#             80,
#             90,
#             100
#         ],
#         dtype=torch.long
#     )

#     # --------------------------------------------------
#     # Create dataset
#     # --------------------------------------------------

#     dataset = GenieeTextDataset(
#         token_ids=token_ids,
#         max_seq_len=4,
#         stride=1
#     )

#     # --------------------------------------------------
#     # DataLoader
#     # --------------------------------------------------

#     dataloader = DataLoader(
#         dataset,
#         batch_size=1,
#         shuffle=False
#     )

#     # --------------------------------------------------
#     # First batch
#     # --------------------------------------------------

#     input_ids, target_ids = next(
#         iter(dataloader)
#     )

#     # --------------------------------------------------
#     # Remove batch dimension
#     # --------------------------------------------------

#     input_ids = input_ids[0]

#     target_ids = target_ids[0]

#     # --------------------------------------------------
#     # Validate
#     # --------------------------------------------------

#     assert torch.equal(
#         input_ids,
#         torch.tensor(
#             [10, 20, 30, 40]
#         )
#     )

#     assert torch.equal(
#         target_ids,
#         torch.tensor(
#             [20, 30, 40, 50]
#         )
#     )

#     # --------------------------------------------------
#     # Every target must be the next token
#     # --------------------------------------------------

#     assert torch.equal(
#         target_ids,
#         input_ids.roll(-1)
#     )


# def test_dataloader_multiple_batches():

#     # --------------------------------------------------
#     # Create token IDs
#     # --------------------------------------------------

#     token_ids = torch.arange(
#         100,
#         dtype=torch.long
#     )

#     # --------------------------------------------------
#     # Dataset
#     # --------------------------------------------------

#     dataset = GenieeTextDataset(
#         token_ids=token_ids,
#         max_seq_len=10,
#         stride=10
#     )

#     # --------------------------------------------------
#     # DataLoader
#     # --------------------------------------------------

#     dataloader = DataLoader(
#         dataset,
#         batch_size=2,
#         shuffle=False
#     )

#     # --------------------------------------------------
#     # Count batches
#     # --------------------------------------------------

#     number_of_batches = 0

#     for input_ids, target_ids in dataloader:

#         number_of_batches += 1

#         # ----------------------------------------------
#         # Validate dimensions
#         # ----------------------------------------------

#         assert input_ids.ndim == 2

#         assert target_ids.ndim == 2

#         # ----------------------------------------------
#         # Input and target must have same shape
#         # ----------------------------------------------

#         assert input_ids.shape == target_ids.shape

#         # ----------------------------------------------
#         # Sequence length
#         # ----------------------------------------------

#         assert input_ids.shape[1] == 10

#     print()

#     print(
#         "Number of batches:",
#         number_of_batches
#     )

#     assert number_of_batches > 0