from __future__ import annotations

from pathlib import Path
import sys

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tokenizer.tokenizer import GenieeTokenizer
from data.pretraining_dataset import PretrainingDataset


# ============================================================
# PATHS
# ============================================================

TOKENIZER_PATH = (
    PROJECT_ROOT
    / "tokenizer"
    / "artifacts"
    / "geniee.model"
)

TRAIN_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "splits"
    / "train.txt"
)

VALIDATION_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "splits"
    / "validation.txt"
)

MAX_SEQ_LEN = 128


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 70)
    print("GENIEE PRETRAINING DATASET INSPECTION")
    print("=" * 70)

    print()
    print(f"Tokenizer : {TOKENIZER_PATH}")
    print(f"Train     : {TRAIN_PATH}")
    print(f"Validation: {VALIDATION_PATH}")
    print(f"Max seq   : {MAX_SEQ_LEN}")

    # --------------------------------------------------------
    # Tokenizer
    # --------------------------------------------------------

    tokenizer = GenieeTokenizer(
        model_path=TOKENIZER_PATH
    )

    print()
    print(f"Vocabulary size : {tokenizer.vocab_size}")

    # --------------------------------------------------------
    # Train dataset
    # --------------------------------------------------------

    train_dataset = PretrainingDataset(
        text_path=TRAIN_PATH,
        tokenizer=tokenizer,
        max_seq_len=MAX_SEQ_LEN,
    )

    # --------------------------------------------------------
    # Validation dataset
    # --------------------------------------------------------

    validation_dataset = PretrainingDataset(
        text_path=VALIDATION_PATH,
        tokenizer=tokenizer,
        max_seq_len=MAX_SEQ_LEN,
    )

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    print()
    print("-" * 70)
    print("DATASET STATISTICS")
    print("-" * 70)

    print()
    print("TRAIN")
    print(f"Tokens    : {train_dataset.total_tokens:,}")
    print(f"Sequences : {train_dataset.num_sequences:,}")

    print()
    print("VALIDATION")
    print(f"Tokens    : {validation_dataset.total_tokens:,}")
    print(f"Sequences : {validation_dataset.num_sequences:,}")

    # --------------------------------------------------------
    # First training sample
    # --------------------------------------------------------

    sample = train_dataset[0]

    input_ids = sample["input_ids"]
    target_ids = sample["target_ids"]

    print()
    print("-" * 70)
    print("FIRST TRAINING SAMPLE")
    print("-" * 70)

    print()
    print("Input IDs:")
    print(input_ids.tolist())

    print()
    print("Target IDs:")
    print(target_ids.tolist())

    print()
    print(
        f"Input shape  : {tuple(input_ids.shape)}"
    )

    print(
        f"Target shape : {tuple(target_ids.shape)}"
    )

    # --------------------------------------------------------
    # Verify next-token alignment
    # --------------------------------------------------------

    alignment_ok = torch.equal(
        input_ids[1:],
        target_ids[:-1],
    )

    print()
    print(
        f"Next-token alignment: {alignment_ok}"
    )

    # --------------------------------------------------------
    # Decode sample
    # --------------------------------------------------------

    decoded_input = tokenizer.decode(
        input_ids.tolist()
    )

    decoded_target = tokenizer.decode(
        target_ids.tolist()
    )

    print()
    print("-" * 70)
    print("DECODED SAMPLE")
    print("-" * 70)

    print()
    print("INPUT:")
    print(decoded_input)

    print()
    print("TARGET:")
    print(decoded_target)

    print()
    print("=" * 70)
    print("INSPECTION COMPLETE")
    print("=" * 70)
    print()


if __name__ == "__main__":
    main()