from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import Dataset

from tokenizer.tokenizer import GenieeTokenizer


class PretrainingDataset(Dataset):
    """
    Dataset for decoder-only language-model pretraining.

    The dataset:
        1. Reads plain text.
        2. Tokenizes the complete corpus.
        3. Splits tokens into fixed-length sequences.
        4. Creates input/target pairs for next-token prediction.

    Example:

        Tokens:
            A B C D E F

        input_ids:
            A B C D E

        target_ids:
            B C D E F
    """

    def __init__(
        self,
        text_path: str | Path,
        tokenizer: GenieeTokenizer,
        max_seq_len: int,
    ):
        self.text_path = Path(text_path)
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

        if not self.text_path.exists():
            raise FileNotFoundError(
                f"Corpus file not found: {self.text_path}"
            )

        if self.max_seq_len <= 0:
            raise ValueError(
                "max_seq_len must be greater than 0"
            )

        # --------------------------------------------------
        # Read corpus
        # --------------------------------------------------

        text = self.text_path.read_text(
            encoding="utf-8"
        ).strip()

        if not text:
            raise ValueError(
                f"Corpus is empty: {self.text_path}"
            )

        self.text = text

        # --------------------------------------------------
        # Tokenize corpus
        # --------------------------------------------------

        self.token_ids = tokenizer.encode(
            text,
            add_bos=True,
            add_eos=True,
        )

        if len(self.token_ids) < 2:
            raise ValueError(
                "Corpus produced fewer than 2 tokens."
            )

        # --------------------------------------------------
        # Create fixed-length sequences
        # --------------------------------------------------

        self.samples = []

        # We need max_seq_len + 1 tokens:
        #
        # input  = tokens[0 : max_seq_len]
        # target = tokens[1 : max_seq_len + 1]
        #
        # Therefore, the final usable sequence starts
        # at max_seq_len intervals.

        sequence_length = self.max_seq_len + 1

        for start in range(
            0,
            len(self.token_ids) - sequence_length + 1,
            self.max_seq_len,
        ):
            chunk = self.token_ids[
                start:start + sequence_length
            ]

            if len(chunk) < sequence_length:
                continue

            input_ids = chunk[:-1]
            target_ids = chunk[1:]

            self.samples.append(
                (
                    torch.tensor(
                        input_ids,
                        dtype=torch.long,
                    ),
                    torch.tensor(
                        target_ids,
                        dtype=torch.long,
                    ),
                )
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        input_ids, target_ids = self.samples[index]

        return {
            "input_ids": input_ids,
            "target_ids": target_ids,
        }

    @property
    def total_tokens(self) -> int:
        """Total number of tokens in the corpus."""
        return len(self.token_ids)

    @property
    def num_sequences(self) -> int:
        """Number of complete training sequences."""
        return len(self.samples)