# from pathlib import Path
# import sys
# PROJECT_ROOT = Path(__file__).resolve().parent.parent
# sys.path.insert(0, str(PROJECT_ROOT))
# from data.corpus import GenieeCorpus
# from data.build_dataset import GenieeDataModule


# def main():

#     print()
#     print("=" * 60)
#     print("GENIEE DATASET INSPECTION")
#     print("=" * 60)


#     # ==========================================================
#     # Corpus
#     # ==========================================================

#     corpus = GenieeCorpus(
#         corpus_dir="corpus",
#         validation_split=0.1,
#         seed=42
#     )


#     (
#         train_documents,
#         validation_documents
#     ) = corpus.train_validation_documents()


#     # ==========================================================
#     # Document split
#     # ==========================================================

#     print()
#     print("=" * 60)
#     print("DOCUMENT SPLIT")
#     print("=" * 60)


#     print()
#     print(
#         f"Training documents   : "
#         f"{len(train_documents)}"
#     )

#     print(
#         f"Validation documents : "
#         f"{len(validation_documents)}"
#     )


#     print()
#     print("TRAINING DOCUMENTS")
#     print("-" * 60)


#     for document in train_documents:

#         print(
#             f"[{document['category']:<15}] "
#             f"{document['path']}"
#         )


#     print()
#     print("VALIDATION DOCUMENTS")
#     print("-" * 60)


#     for document in validation_documents:

#         print(
#             f"[{document['category']:<15}] "
#             f"{document['path']}"
#         )


#     # ==========================================================
#     # Instruction documents
#     # ==========================================================

#     print()
#     print("=" * 60)
#     print("INSTRUCTION DOCUMENT CHECK")
#     print("=" * 60)


#     training_instruction_documents = [
#         document
#         for document in train_documents
#         if document["category"] == "instructions"
#     ]


#     validation_instruction_documents = [
#         document
#         for document in validation_documents
#         if document["category"] == "instructions"
#     ]


#     print()
#     print(
#         "Instruction documents in training : "
#         f"{len(training_instruction_documents)}"
#     )

#     print(
#         "Instruction documents in validation : "
#         f"{len(validation_instruction_documents)}"
#     )


#     print()
#     print("TRAINING INSTRUCTION FILES")
#     print("-" * 60)


#     for document in training_instruction_documents:

#         print(
#             document["path"]
#         )


#     print()
#     print("VALIDATION INSTRUCTION FILES")
#     print("-" * 60)


#     for document in validation_instruction_documents:

#         print(
#             document["path"]
#         )


#     # ==========================================================
#     # Build dataset
#     # ==========================================================

#     print()
#     print("=" * 60)
#     print("TOKENIZED DATASET")
#     print("=" * 60)


#     # data_module = GenieeDataModule(
#     #     corpus_dir="corpus",
#     #     tokenizer_path="tokenizer/artifacts/geniee.model",
#     #     max_seq_len=128,
#     #     stride=64,
#     #     validation_split=0.1
#     # )
#     # data_module = GenieeDataModule(
#     #     corpus_dir="corpus",
#     #     tokenizer_path="tokenizer/artifacts/geniee.model",
#     #     max_seq_len=128,
#     #     batch_size=2,
#     #     stride=None,
#     #     validation_split=0.2,
#     #     shuffle_train=True
#     # )
#     data_module = GenieeDataModule(
#         corpus_dir="corpus",
#         tokenizer_path="tokenizer/artifacts/geniee.model",
#         max_seq_len=128,
#         batch_size=2,
#         stride=None,
#         validation_split=0.2,
#         shuffle_train=True
#     )


#     train_dataset, validation_dataset = (
#         data_module.build_datasets()
#     )


#     print()
#     print(
#         f"Train tokens        : "
#         f"{len(data_module.train_token_ids):,}"
#     )

#     print(
#         f"Validation tokens   : "
#         f"{len(data_module.validation_token_ids):,}"
#     )

#     print(
#         f"Train sequences     : "
#         f"{len(train_dataset):,}"
#     )

#     print(
#         f"Validation sequences: "
#         f"{len(validation_dataset):,}"
#     )


#     # ==========================================================
#     # Find an actual instruction example
#     # ==========================================================

#     print()
#     print("=" * 60)
#     print("INSTRUCTION SAMPLE")
#     print("=" * 60)


#     instruction_example_found = False


#     for document in train_documents:

#         if document["category"] != "instructions":
#             continue


#         print()
#         print(
#             f"Source: {document['path']}"
#         )


#         print()
#         print("Raw text:")
#         print("-" * 60)


#         # Print only the beginning so the output
#         # remains manageable.

#         print(
#             document["text"][:1500]
#         )


#         instruction_example_found = True

#         break


#     if not instruction_example_found:

#         print()
#         print(
#             "WARNING: No instruction document "
#             "was assigned to training."
#         )


#     print()
#     print("=" * 60)
#     print("DATASET INSPECTION COMPLETE")
#     print("=" * 60)


# if __name__ == "__main__":

#     main()

from pathlib import Path
import sys

import torch


# ==========================================================
# Project root
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ==========================================================
# Project imports
# ==========================================================

from data.corpus import GenieeCorpus
from data.build_dataset import GenieeDataModule


# ==========================================================
# Configuration
# ==========================================================

CORPUS_DIR = "corpus"
TOKENIZER_PATH = "tokenizer/artifacts/geniee.model"

MAX_SEQ_LEN = 128
BATCH_SIZE = 2

# Independent examples are used now.
STRIDE = None

VALIDATION_SPLIT = 0.2
SEED = 42

PAD_ID = 0
BOS_ID = 2
EOS_ID = 3


# ==========================================================
# Helpers
# ==========================================================

def count_real_tokens(dataset):
    """
    Count non-padding input tokens.
    """

    total = 0

    for input_ids, target_ids in dataset:

        total += int(
            (input_ids != PAD_ID).sum().item()
        )

    return total


def inspect_sample(
    dataset,
    tokenizer,
    sample_index=0
):
    """
    Inspect one actual input/target training pair.
    """

    input_ids, target_ids = dataset[sample_index]

    print()
    print("=" * 60)
    print("ACTUAL DATASET SAMPLE")
    print("=" * 60)

    print()
    print(f"Sample index : {sample_index}")

    print()
    print("Input IDs:")
    print(input_ids.tolist())

    print()
    print("Target IDs:")
    print(target_ids.tolist())

    real_input_length = int(
        (input_ids != PAD_ID).sum().item()
    )

    padding_count = int(
        (input_ids == PAD_ID).sum().item()
    )

    print()
    print(f"Real input tokens : {real_input_length}")
    print(f"Padding tokens    : {padding_count}")

    # ------------------------------------------------------
    # Decode input
    # ------------------------------------------------------

    real_input_ids = input_ids[
        input_ids != PAD_ID
    ].tolist()

    real_target_ids = target_ids[
        target_ids != PAD_ID
    ].tolist()

    print()
    print("Decoded input:")
    print("-" * 60)

    print(
        tokenizer.decode(real_input_ids)
    )

    print()
    print("Decoded target:")
    print("-" * 60)

    print(
        tokenizer.decode(real_target_ids)
    )

    # ------------------------------------------------------
    # Verify causal shift
    # ------------------------------------------------------

    print()
    print("SHIFT CHECK")
    print("-" * 60)

    input_without_padding = input_ids[
        input_ids != PAD_ID
    ]

    target_without_padding = target_ids[
        target_ids != PAD_ID
    ]

    if len(input_without_padding) > 0:

        expected_target = input_without_padding[1:]

        actual_target = target_without_padding[:-1]

        shift_is_correct = torch.equal(
            expected_target,
            actual_target
        )

        print(
            f"Input → Target shift correct : "
            f"{shift_is_correct}"
        )

    # ------------------------------------------------------
    # Special token check
    # ------------------------------------------------------

    print()
    print("SPECIAL TOKEN CHECK")
    print("-" * 60)

    if real_input_ids:

        print(
            f"First input token : "
            f"{real_input_ids[0]}"
        )

        print(
            f"Expected BOS ID  : "
            f"{BOS_ID}"
        )

    if real_target_ids:

        print(
            f"Last target token : "
            f"{real_target_ids[-1]}"
        )

        print(
            f"Expected EOS ID  : "
            f"{EOS_ID}"
        )


# ==========================================================
# Main
# ==========================================================

def main():

    print()
    print("=" * 60)
    print("GENIEE DATASET INSPECTION")
    print("=" * 60)

    # ======================================================
    # Corpus
    # ======================================================

    corpus = GenieeCorpus(
        corpus_dir=CORPUS_DIR,
        validation_split=VALIDATION_SPLIT,
        seed=SEED
    )

    (
        train_documents,
        validation_documents
    ) = corpus.train_validation_documents()

    # ======================================================
    # Document split
    # ======================================================

    print()
    print("=" * 60)
    print("DOCUMENT SPLIT")
    print("=" * 60)

    print()
    print(
        f"Training documents   : "
        f"{len(train_documents)}"
    )

    print(
        f"Validation documents : "
        f"{len(validation_documents)}"
    )

    # ------------------------------------------------------
    # Training categories
    # ------------------------------------------------------

    print()
    print("TRAINING DOCUMENT CATEGORIES")
    print("-" * 60)

    training_categories = {}

    for document in train_documents:

        category = document["category"]

        training_categories[category] = (
            training_categories.get(category, 0) + 1
        )

    for category, count in sorted(
        training_categories.items()
    ):

        print(
            f"{category:<20}: {count}"
        )

    # ------------------------------------------------------
    # Validation categories
    # ------------------------------------------------------

    print()
    print("VALIDATION DOCUMENT CATEGORIES")
    print("-" * 60)

    validation_categories = {}

    for document in validation_documents:

        category = document["category"]

        validation_categories[category] = (
            validation_categories.get(category, 0) + 1
        )

    for category, count in sorted(
        validation_categories.items()
    ):

        print(
            f"{category:<20}: {count}"
        )

    # ======================================================
    # Instruction documents
    # ======================================================

    print()
    print("=" * 60)
    print("INSTRUCTION DOCUMENT CHECK")
    print("=" * 60)

    training_instruction_documents = [
        document
        for document in train_documents
        if document["category"] == "instructions"
    ]

    validation_instruction_documents = [
        document
        for document in validation_documents
        if document["category"] == "instructions"
    ]

    print()
    print(
        "Instruction documents in training   : "
        f"{len(training_instruction_documents)}"
    )

    print(
        "Instruction documents in validation : "
        f"{len(validation_instruction_documents)}"
    )

    # ------------------------------------------------------
    # Instruction files
    # ------------------------------------------------------

    print()
    print("TRAINING INSTRUCTION FILES")
    print("-" * 60)

    for document in training_instruction_documents:

        print(
            document["path"]
        )

    print()
    print("VALIDATION INSTRUCTION FILES")
    print("-" * 60)

    for document in validation_instruction_documents:

        print(
            document["path"]
        )

    # ======================================================
    # Build dataset
    # ======================================================

    print()
    print("=" * 60)
    print("TOKENIZED DATASET")
    print("=" * 60)

    data_module = GenieeDataModule(
        corpus_dir=CORPUS_DIR,
        tokenizer_path=TOKENIZER_PATH,
        max_seq_len=MAX_SEQ_LEN,
        batch_size=BATCH_SIZE,
        stride=STRIDE,
        validation_split=VALIDATION_SPLIT,
        shuffle_train=True
    )

    (
        train_dataset,
        validation_dataset
    ) = data_module.build_datasets()

    # ======================================================
    # Correct token statistics
    # ======================================================

    train_real_tokens = count_real_tokens(
        train_dataset
    )

    validation_real_tokens = count_real_tokens(
        validation_dataset
    )

    print()
    print(
        f"Train examples             : "
        f"{len(data_module.train_token_ids):,}"
    )

    print(
        f"Validation examples        : "
        f"{len(data_module.validation_token_ids):,}"
    )

    print(
        f"Train dataset sequences    : "
        f"{len(train_dataset):,}"
    )

    print(
        f"Validation dataset sequences: "
        f"{len(validation_dataset):,}"
    )

    print()
    print(
        f"Actual train tokens        : "
        f"{train_real_tokens:,}"
    )

    print(
        f"Actual validation tokens   : "
        f"{validation_real_tokens:,}"
    )

    # ======================================================
    # Padding statistics
    # ======================================================

    train_total_positions = (
        len(train_dataset) * MAX_SEQ_LEN
    )

    validation_total_positions = (
        len(validation_dataset) * MAX_SEQ_LEN
    )

    train_padding = (
        train_total_positions
        - train_real_tokens
    )

    validation_padding = (
        validation_total_positions
        - validation_real_tokens
    )

    print()
    print("PADDING STATISTICS")
    print("-" * 60)

    print(
        f"Train padding tokens       : "
        f"{train_padding:,}"
    )

    print(
        f"Validation padding tokens  : "
        f"{validation_padding:,}"
    )

    if train_total_positions > 0:

        print(
            f"Train padding percentage   : "
            f"{train_padding / train_total_positions * 100:.2f}%"
        )

    if validation_total_positions > 0:

        print(
            f"Validation padding percentage: "
            f"{validation_padding / validation_total_positions * 100:.2f}%"
        )

    # ======================================================
    # Instruction sample
    # ======================================================

    print()
    print("=" * 60)
    print("INSTRUCTION SAMPLE")
    print("=" * 60)

    instruction_example_found = False

    for document in train_documents:

        if document["category"] != "instructions":
            continue

        print()
        print(
            f"Source: {document['path']}"
        )

        print()
        print("Raw text:")
        print("-" * 60)

        print(
            document["text"][:1500]
        )

        instruction_example_found = True

        break

    if not instruction_example_found:

        print()
        print(
            "WARNING: No instruction document "
            "was assigned to training."
        )

    # ======================================================
    # Inspect actual sequence
    # ======================================================

    if len(train_dataset) > 0:

        inspect_sample(
            train_dataset,
            data_module.tokenizer,
            sample_index=0
        )

    # ======================================================
    # Final validation
    # ======================================================

    print()
    print("=" * 60)
    print("FINAL DATASET CHECK")
    print("=" * 60)

    checks_passed = True

    # ------------------------------------------------------
    # Check 1
    # ------------------------------------------------------

    if len(train_dataset) == 0:

        print(
            "FAIL: Training dataset is empty."
        )

        checks_passed = False

    else:

        print(
            "PASS: Training dataset contains samples."
        )

    # ------------------------------------------------------
    # Check 2
    # ------------------------------------------------------

    if len(validation_dataset) == 0:

        print(
            "FAIL: Validation dataset is empty."
        )

        checks_passed = False

    else:

        print(
            "PASS: Validation dataset contains samples."
        )

    # ------------------------------------------------------
    # Check 3
    # ------------------------------------------------------

    if train_real_tokens == 0:

        print(
            "FAIL: Training dataset contains "
            "zero real tokens."
        )

        checks_passed = False

    else:

        print(
            "PASS: Training dataset contains "
            "real tokens."
        )

    # ------------------------------------------------------
    # Check 4
    # ------------------------------------------------------

    if len(training_instruction_documents) == 0:

        print(
            "FAIL: No instruction documents "
            "in training."
        )

        checks_passed = False

    else:

        print(
            "PASS: Instruction documents "
            "are present in training."
        )

    # ------------------------------------------------------
    # Check 5
    # ------------------------------------------------------

    if (
        len(validation_instruction_documents) == 0
    ):

        print(
            "WARNING: No instruction documents "
            "in validation."
        )

    else:

        print(
            "PASS: Instruction documents "
            "are present in validation."
        )

    # ------------------------------------------------------
    # Final status
    # ------------------------------------------------------

    print()

    if checks_passed:

        print(
            "OVERALL RESULT: DATASET BASIC CHECKS PASSED"
        )

    else:

        print(
            "OVERALL RESULT: DATASET CHECKS FAILED"
        )

    print()
    print("=" * 60)
    print("DATASET INSPECTION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":

    main()