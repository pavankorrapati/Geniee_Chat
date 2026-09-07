from pathlib import Path
import sys

import torch


# ==========================================================
# Project root
# ==========================================================

PROJECT_ROOT = (
    Path(__file__).resolve().parent.parent
)

# ----------------------------------------------------------
# Make project root available for imports when this file
# is executed directly with:
#
# python training/train.py
# ----------------------------------------------------------

if str(PROJECT_ROOT) not in sys.path:

    sys.path.insert(
        0,
        str(PROJECT_ROOT)
    )


# ==========================================================
# Geniee imports
# ==========================================================

from data.build_dataset import (
    GenieeDataModule
)

from model.config import (
    GenieeConfig
)

from model.geniee_model import (
    GenieeModel
)

from training.trainer import (
    GenieeTrainer
)


# ==========================================================
# Configuration
# ==========================================================

CORPUS_DIR = (
    PROJECT_ROOT / "corpus"
)

TOKENIZER_PATH = (
    PROJECT_ROOT
    / "tokenizer"
    / "artifacts"
    / "geniee.model"
)

CHECKPOINT_DIR = (
    PROJECT_ROOT / "checkpoints"
)


# ==========================================================
# Training configuration
# ==========================================================

MAX_SEQ_LEN = 128

BATCH_SIZE = 2

STRIDE = None

VALIDATION_SPLIT = 0.2

LEARNING_RATE = 3e-4

WEIGHT_DECAY = 0.01

MAX_GRAD_NORM = 1.0

EPOCHS = 5

SAVE_EVERY = 1


# ==========================================================
# Device
# ==========================================================

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ==========================================================
# Build DataModule
# ==========================================================

def build_data_module():

    print()
    print(
        "Building Geniee data module..."
    )

    # data_module = GenieeDataModule(

    #     corpus_dir=CORPUS_DIR,

    #     tokenizer_path=TOKENIZER_PATH,

    #     max_seq_len=MAX_SEQ_LEN,

    #     batch_size=BATCH_SIZE,

    #     stride=STRIDE,

    #     validation_split=VALIDATION_SPLIT,

    #     shuffle_train=True
    # )
    data_module = GenieeDataModule(corpus_dir=CORPUS_DIR,tokenizer_path=TOKENIZER_PATH,max_seq_len=MAX_SEQ_LEN,batch_size=BATCH_SIZE,stride=STRIDE,validation_split=VALIDATION_SPLIT,shuffle_train=True,seed=42)

    return data_module


# ==========================================================
# Build Model
# ==========================================================

def build_model(
    vocab_size
):

    print()
    print(
        "Building Geniee model..."
    )

    # ------------------------------------------------------
    # IMPORTANT
    #
    # Use the existing GenieeConfig API.
    # Do not introduce configuration field names here that
    # don't already exist in model/config.py.
    # ------------------------------------------------------

    config = GenieeConfig(
        vocab_size=vocab_size,
        max_seq_len=MAX_SEQ_LEN
    )

    model = GenieeModel(
        config
    )

    return model


# ==========================================================
# Print model information
# ==========================================================

def print_model_info(
    model
):

    total_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    trainable_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    print()
    print(
        "=" * 60
    )

    print(
        "GENIEE MODEL"
    )

    print(
        "=" * 60
    )

    print(
        f"Total parameters     : "
        f"{total_parameters:,}"
    )

    print(
        f"Trainable parameters : "
        f"{trainable_parameters:,}"
    )

    print(
        f"Device               : "
        f"{DEVICE}"
    )

    print(
        "=" * 60
    )


# ==========================================================
# Main training function
# ==========================================================

def main():

    print()
    print(
        "=" * 60
    )

    print(
        "GENIEE LANGUAGE MODEL"
    )

    print(
        "Training Pipeline"
    )

    print(
        "=" * 60
    )

    # ------------------------------------------------------
    # Verify tokenizer
    # ------------------------------------------------------

    if not TOKENIZER_PATH.exists():

        raise FileNotFoundError(
            "Geniee tokenizer was not found:\n"
            f"{TOKENIZER_PATH}"
        )

    # ------------------------------------------------------
    # Verify corpus
    # ------------------------------------------------------

    if not CORPUS_DIR.exists():

        raise FileNotFoundError(
            "Geniee corpus directory was not found:\n"
            f"{CORPUS_DIR}"
        )

    # ------------------------------------------------------
    # Build DataModule
    # ------------------------------------------------------

    data_module = (
        build_data_module()
    )

    # ------------------------------------------------------
    # Build DataLoaders
    # ------------------------------------------------------

    (
        train_loader,
        validation_loader
    ) = data_module.build()

    # ------------------------------------------------------
    # Print data information
    # ------------------------------------------------------

    data_module.summary()

    # ------------------------------------------------------
    # Determine vocabulary size
    # ------------------------------------------------------

    vocab_size = (
        data_module.tokenizer.vocab_size
    )

    # print()
    # print(
    #     f"Vocabulary size: {vocab_size}"
    # )

    # # ------------------------------------------------------
    # # Build model
    # # ------------------------------------------------------

    # model = build_model(
    #     vocab_size=vocab_size
    # )
    print()
    print(
        f"Vocabulary size: {vocab_size}"
    )

    # ------------------------------------------------------
    # Verify expected vocabulary size
    # ------------------------------------------------------

    EXPECTED_VOCAB_SIZE = 2048

    if vocab_size != EXPECTED_VOCAB_SIZE:

        raise ValueError(
            f"Unexpected tokenizer vocabulary size: "
            f"{vocab_size}. "
            f"Expected: {EXPECTED_VOCAB_SIZE}"
        )

    print(
        f"Tokenizer vocabulary verified: "
        f"{vocab_size}"
    )

    # ------------------------------------------------------
    # Build model
    # ------------------------------------------------------

    model = build_model(
        vocab_size=vocab_size
    )

    # ------------------------------------------------------
    # Model information
    # ------------------------------------------------------

    print_model_info(
        model
    )

    # ------------------------------------------------------
    # Create trainer
    # ------------------------------------------------------

    trainer = GenieeTrainer(

        model=model,

        train_loader=train_loader,

        validation_loader=validation_loader,

        learning_rate=LEARNING_RATE,

        weight_decay=WEIGHT_DECAY,

        max_grad_norm=MAX_GRAD_NORM,

        device=DEVICE,

        checkpoint_dir=CHECKPOINT_DIR
    )

    # ------------------------------------------------------
    # Start training
    # ------------------------------------------------------

    # trainer.train(

    #     epochs=EPOCHS,

    #     save_every=SAVE_EVERY
    # )
    trainer.train(epochs=EPOCHS,save_every=SAVE_EVERY,early_stopping_patience=2)

    # ------------------------------------------------------
    # Final summary
    # ------------------------------------------------------

    print()
    print(
        "=" * 60
    )

    print(
        "GENIEE TRAINING FINISHED"
    )

    print(
        "=" * 60
    )

    print(
        f"Final training loss : "
        f"{trainer.train_losses[-1]:.6f}"
    )

    print(
        f"Final validation loss : "
        f"{trainer.validation_losses[-1]:.6f}"
    )

    print(
        f"Global steps : "
        f"{trainer.global_step}"
    )

    print(
        f"Checkpoints : "
        f"{CHECKPOINT_DIR}"
    )

    print(
        "=" * 60
    )


# ==========================================================
# Entry point
# ==========================================================

if __name__ == "__main__":

    main()