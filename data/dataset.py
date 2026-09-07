# import torch
# from torch.utils.data import Dataset


# class GenieeTextDataset(Dataset):
#     """
#     Dataset for causal language modeling.

#     Converts a sequence of token IDs into
#     input/target pairs for next-token prediction.

#     Example:

#         tokens:
#         [10, 20, 30, 40, 50]

#         max_seq_len = 128

#         input:
#         [10, 20, 30, 40]

#         target:
#         [20, 30, 40, 50]
#     """

#     def __init__(
#         self,
#         token_ids,
#         max_seq_len,
#         stride=None
#     ):
#         super().__init__()

#         # --------------------------------------------------
#         # Validate input
#         # --------------------------------------------------

#         if not isinstance(
#             token_ids,
#             torch.Tensor
#         ):
#             token_ids = torch.tensor(
#                 token_ids,
#                 dtype=torch.long
#             )

#         if token_ids.ndim != 1:
#             raise ValueError(
#                 "token_ids must be a 1-dimensional tensor"
#             )

#         if token_ids.dtype != torch.long:
#             token_ids = token_ids.long()

#         if max_seq_len <= 0:
#             raise ValueError(
#                 "max_seq_len must be greater than 0"
#             )

#         if len(token_ids) <= max_seq_len:
#             raise ValueError(
#                 "token_ids must contain more tokens "
#                 "than max_seq_len"
#             )

#         # --------------------------------------------------
#         # Store values
#         # --------------------------------------------------

#         self.token_ids = token_ids
#         self.max_seq_len = max_seq_len

#         # --------------------------------------------------
#         # Default stride
#         #
#         # stride = max_seq_len
#         # means non-overlapping chunks.
#         # --------------------------------------------------

#         if stride is None:
#             stride = max_seq_len

#         if stride <= 0:
#             raise ValueError(
#                 "stride must be greater than 0"
#             )

#         self.stride = stride

#         # --------------------------------------------------
#         # Calculate number of samples
#         # --------------------------------------------------

#         self.num_samples = (len(self.token_ids)- self.max_seq_len- 1) // self.stride + 1

#     def __len__(self):

#         return self.num_samples

#     def __getitem__(self, index):

#         if index < 0 or index >= self.num_samples:
#             raise IndexError(
#                 "Dataset index out of range"
#             )

#         # --------------------------------------------------
#         # Start position
#         # --------------------------------------------------

#         start = index * self.stride

#         # --------------------------------------------------
#         # Input sequence
#         # --------------------------------------------------

#         input_ids = self.token_ids[
#             start:
#             start + self.max_seq_len
#         ]

#         # --------------------------------------------------
#         # Target sequence
#         #
#         # Shifted one token to the right.
#         # --------------------------------------------------

#         target_ids = self.token_ids[
#             start + 1:
#             start + self.max_seq_len + 1
#         ]

#         return input_ids, target_ids

import torch
from torch.utils.data import Dataset


class GenieeTextDataset(Dataset):
    """
    Dataset for causal language modeling.

    Each training example is kept independent.

    The dataset:
        1. Creates input/target pairs
        2. Pads shorter examples
        3. Masks padding through PAD targets
        4. Never joins unrelated documents together
    """

    def __init__(
        self,
        sequences,
        max_seq_len,
        pad_id=0
    ):
        super().__init__()

        if not sequences:
            raise ValueError(
                "No sequences were provided to GenieeTextDataset."
            )

        if max_seq_len <= 0:
            raise ValueError(
                "max_seq_len must be greater than 0."
            )

        if pad_id < 0:
            raise ValueError(
                "pad_id must be >= 0."
            )

        self.max_seq_len = max_seq_len
        self.pad_id = pad_id

        # --------------------------------------------------
        # Prepare independent samples
        # --------------------------------------------------

        self.samples = []

        for sequence in sequences:

            if not isinstance(sequence, torch.Tensor):

                sequence = torch.tensor(
                    sequence,
                    dtype=torch.long
                )

            if sequence.ndim != 1:

                raise ValueError(
                    "Every sequence must be 1-dimensional."
                )

            sequence = sequence.long()

            # Need at least:
            #
            # input  = token 1
            # target = token 2
            #
            if len(sequence) < 2:
                continue

            # --------------------------------------------------
            # If sequence fits into context
            # --------------------------------------------------

            if len(sequence) <= max_seq_len + 1:

                input_ids = sequence[:-1]

                target_ids = sequence[1:]

                self.samples.append(
                    self._pad_sample(
                        input_ids,
                        target_ids
                    )
                )

            # --------------------------------------------------
            # Long document
            #
            # Split ONLY inside this document.
            # Never cross into another document.
            # --------------------------------------------------

            else:

                start = 0

                while start < len(sequence) - 1:

                    chunk = sequence[
                        start:
                        start + max_seq_len + 1
                    ]

                    if len(chunk) < 2:
                        break

                    input_ids = chunk[:-1]

                    target_ids = chunk[1:]

                    self.samples.append(
                        self._pad_sample(
                            input_ids,
                            target_ids
                        )
                    )

                    start += max_seq_len

        if not self.samples:

            raise ValueError(
                "No valid training samples were created."
            )

    # ======================================================
    # Padding
    # ======================================================

    def _pad_sample(
        self,
        input_ids,
        target_ids
    ):

        input_ids = input_ids[:self.max_seq_len]

        target_ids = target_ids[:self.max_seq_len]

        input_length = len(input_ids)

        padding_length = (
            self.max_seq_len -
            input_length
        )

        if padding_length > 0:

            input_padding = torch.full(
                (
                    padding_length,
                ),
                self.pad_id,
                dtype=torch.long
            )

            target_padding = torch.full(
                (
                    padding_length,
                ),
                self.pad_id,
                dtype=torch.long
            )

            input_ids = torch.cat(
                [
                    input_ids,
                    input_padding
                ]
            )

            target_ids = torch.cat(
                [
                    target_ids,
                    target_padding
                ]
            )

        return (
            input_ids,
            target_ids
        )

    # ======================================================
    # Length
    # ======================================================

    def __len__(self):

        return len(self.samples)

    # ======================================================
    # Get sample
    # ======================================================

    def __getitem__(
        self,
        index
    ):

        if index < 0 or index >= len(self.samples):

            raise IndexError(
                "Dataset index out of range."
            )

        return self.samples[index]