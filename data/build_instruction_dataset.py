from pathlib import Path
import sys


# ==========================================================
# Project root
# ==========================================================

PROJECT_ROOT = (
    Path(__file__).resolve().parent.parent
)

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT)
    )


from data.instruction_dataset import (
    GenieeInstructionDataset
)

from tokenizer.tokenizer import (
    GenieeTokenizer
)


# ==========================================================
# Paths
# ==========================================================

DATA_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
)

SPLIT_DIR = (
    DATA_DIR
    / "splits"
)

TOKENIZER_PATH = (
    PROJECT_ROOT
    / "tokenizer"
    / "artifacts"
    / "geniee.model"
)


# ==========================================================
# Configuration
# ==========================================================

MAX_SEQ_LEN = 128


# ==========================================================
# Main
# ==========================================================

def main():

    print()
    print("=" * 70)
    print(
        "GENIEE V2 INSTRUCTION DATASET BUILDER"
    )
    print("=" * 70)

    print(
        f"Tokenizer : {TOKENIZER_PATH}"
    )

    print(
        f"Max sequence length : {MAX_SEQ_LEN}"
    )

    print()

    tokenizer = GenieeTokenizer(
        model_path=TOKENIZER_PATH
    )

    print(
        f"Vocabulary size : "
        f"{tokenizer.vocab_size}"
    )

    print(
        f"PAD ID : {tokenizer.pad_id}"
    )

    print(
        f"BOS ID : {tokenizer.bos_id}"
    )

    print(
        f"EOS ID : {tokenizer.eos_id}"
    )

    print()

    split_files = {
        "train": SPLIT_DIR / "train.jsonl",
        "validation": SPLIT_DIR / "validation.jsonl",
        "test": SPLIT_DIR / "test.jsonl",
    }

    datasets = {}

    for split_name, split_path in split_files.items():

        print("-" * 70)

        print(
            f"Building {split_name} dataset"
        )

        print(
            f"Source : {split_path}"
        )

        dataset = GenieeInstructionDataset(
            jsonl_path=split_path,
            tokenizer=tokenizer,
            max_seq_len=MAX_SEQ_LEN,
            pad_id=tokenizer.pad_id,
        )

        datasets[split_name] = dataset

        stats = dataset.statistics()

        print(
            f"Records : "
            f"{stats['records']}"
        )

        print(
            f"Samples : "
            f"{stats['samples']}"
        )

        print(
            f"Sequence tokens : "
            f"{stats['total_sequence_tokens']}"
        )

        print(
            f"Assistant loss tokens : "
            f"{stats['assistant_loss_tokens']}"
        )

        print(
            f"Padding tokens : "
            f"{stats['padding_tokens']}"
        )

        print(
            f"Assistant loss percentage : "
            f"{stats['assistant_loss_percentage']:.2f}%"
        )

        print(
            f"Padding percentage : "
            f"{stats['padding_percentage']:.2f}%"
        )

    # ======================================================
    # Inspect first training sample
    # ======================================================

    print()
    print("=" * 70)
    print("FIRST TRAINING SAMPLE")
    print("=" * 70)

    sample = datasets["train"][0]

    input_ids = sample["input_ids"]
    target_ids = sample["target_ids"]
    loss_mask = sample["loss_mask"]
    attention_mask = sample[
        "attention_mask"
    ]

    print()
    print("Input IDs:")
    print(
        input_ids.tolist()
    )

    print()
    print("Target IDs:")
    print(
        target_ids.tolist()
    )

    print()
    print("Loss mask:")
    print(
        loss_mask.tolist()
    )

    print()
    print("Attention mask:")
    print(
        attention_mask.tolist()
    )

    # ======================================================
    # Decode active input
    # ======================================================

    active_length = int(
        attention_mask.sum().item()
    )

    active_input_ids = (
        input_ids[:active_length]
        .tolist()
    )

    print()
    print("Decoded input:")
    print(
        tokenizer.decode(
            active_input_ids
        )
    )

    # ======================================================
    # Loss mask verification
    # ======================================================

    print()
    print("=" * 70)
    print("LOSS MASK VERIFICATION")
    print("=" * 70)

    assistant_loss_count = int(
        loss_mask.sum().item()
    )

    non_loss_count = int(
        (
            loss_mask == 0
        ).sum().item()
    )

    print(
        f"Assistant loss positions : "
        f"{assistant_loss_count}"
    )

    print(
        f"Non-loss positions : "
        f"{non_loss_count}"
    )

    if assistant_loss_count <= 0:

        raise RuntimeError(
            "ERROR: No assistant tokens "
            "are marked for loss."
        )

    if assistant_loss_count >= active_length:

        print(
            "WARNING: Almost the entire "
            "sequence contributes to loss."
        )

    print()
    print(
        "Instruction dataset build "
        "completed successfully."
    )

    print("=" * 70)


if __name__ == "__main__":
    main()