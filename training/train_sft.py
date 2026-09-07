from __future__ import annotations

from pathlib import Path
import sys
import math
import random
import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset


# ==========================================================
# Project root
# ==========================================================

PROJECT_ROOT = (
    Path(__file__).resolve().parent.parent
)

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ==========================================================
# Geniee imports
# ==========================================================

from model.config import GenieeConfig
from model.geniee_model import GenieeModel

from tokenizer.tokenizer import GenieeTokenizer

from data.instruction_dataset import (
    GenieeInstructionDataset
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

CHECKPOINT_DIR = (
    PROJECT_ROOT
    / "checkpoints"
)

# ----------------------------------------------------------
# Normal SFT checkpoint
# ----------------------------------------------------------

BEST_CHECKPOINT = (
    CHECKPOINT_DIR
    / "geniee_sft_best.pt"
)

# ----------------------------------------------------------
# Debug-overfit checkpoint
#
# IMPORTANT:
# This is separate from the normal SFT checkpoint.
# ----------------------------------------------------------

DEBUG_CHECKPOINT = (
    CHECKPOINT_DIR
    / "geniee_sft_debug_best.pt"
)


# ==========================================================
# Dataset configuration
# ==========================================================

MAX_SEQ_LEN = 128

BATCH_SIZE = 2

NUM_WORKERS = 0


# ==========================================================
# Model configuration
# ==========================================================

D_MODEL = 512

NUM_HEADS = 8

FFN_HIDDEN_DIM = 2048

NUM_LAYERS = 8

DROPOUT = 0.1


# ==========================================================
# Normal training configuration
# ==========================================================

EPOCHS = 10

LEARNING_RATE = 1e-4

WEIGHT_DECAY = 0.01

MAX_GRAD_NORM = 1.0

EARLY_STOPPING_PATIENCE = 3

WARMUP_RATIO = 0.10

SEED = 42


# ==========================================================
# Debug-overfit configuration
# ==========================================================

DEBUG_OVERFIT_SAMPLES = 8

DEBUG_OVERFIT_EPOCHS = 200

DEBUG_OVERFIT_LEARNING_RATE = 3e-4

DEBUG_OVERFIT_BATCH_SIZE = 2

DEBUG_OVERFIT_DROPOUT = 0.0

DEBUG_PRINT_EVERY = 10


# ==========================================================
# Device
# ==========================================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ==========================================================
# Command-line arguments
# ==========================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "Train Geniee using supervised "
            "instruction fine-tuning."
        )
    )

    parser.add_argument(
        "--debug-overfit",
        action="store_true",
        help=(
            "Run a tiny overfit test using "
            "a small subset of the training data."
        ),
    )

    parser.add_argument(
        "--debug-samples",
        type=int,
        default=DEBUG_OVERFIT_SAMPLES,
        help=(
            "Number of samples used by "
            "--debug-overfit."
        ),
    )

    parser.add_argument(
        "--debug-epochs",
        type=int,
        default=DEBUG_OVERFIT_EPOCHS,
        help=(
            "Number of epochs used by "
            "--debug-overfit."
        ),
    )

    return parser.parse_args()


# ==========================================================
# Reproducibility
# ==========================================================

def set_seed(seed: int):

    random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(seed)

    if torch.backends.cudnn.is_available():

        torch.backends.cudnn.deterministic = True

        torch.backends.cudnn.benchmark = False


# ==========================================================
# Parameter count
# ==========================================================

def count_parameters(model):

    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )


# ==========================================================
# Build model
# ==========================================================

def build_model(
    tokenizer,
    dropout=None,
):

    if dropout is None:
        dropout = DROPOUT

    config = GenieeConfig(
        vocab_size=tokenizer.vocab_size,

        max_seq_len=MAX_SEQ_LEN,

        d_model=D_MODEL,

        num_heads=NUM_HEADS,

        ffn_hidden_dim=FFN_HIDDEN_DIM,

        num_layers=NUM_LAYERS,

        dropout=dropout,
    )

    model = GenieeModel(
        config=config
    )

    return model


# ==========================================================
# Assistant-only loss
# ==========================================================

def calculate_sft_loss(
    logits,
    target_ids,
    loss_mask,
):
    """
    Calculate supervised instruction-tuning loss.

    Only positions where:

        loss_mask == 1

    contribute to the loss.

    Returns:

        loss
        masked_loss_sum
        valid_token_count
    """

    batch_size = logits.size(0)

    sequence_length = logits.size(1)

    vocab_size = logits.size(2)

    # ------------------------------------------------------
    # Flatten
    # ------------------------------------------------------

    logits = logits.reshape(
        batch_size
        * sequence_length,
        vocab_size
    )

    target_ids = target_ids.reshape(
        batch_size
        * sequence_length
    )

    loss_mask = loss_mask.reshape(
        batch_size
        * sequence_length
    )

    # ------------------------------------------------------
    # Per-token cross entropy
    # ------------------------------------------------------

    criterion = nn.CrossEntropyLoss(
        reduction="none"
    )

    token_losses = criterion(
        logits,
        target_ids
    )

    # ------------------------------------------------------
    # Assistant-only masking
    # ------------------------------------------------------

    masked_losses = (
        token_losses
        * loss_mask
    )

    # ------------------------------------------------------
    # Valid assistant tokens
    # ------------------------------------------------------

    valid_tokens = loss_mask.sum()

    if valid_tokens.item() == 0:

        raise RuntimeError(
            "Loss mask contains zero "
            "assistant tokens."
        )

    # ------------------------------------------------------
    # Mean loss over assistant tokens
    # ------------------------------------------------------

    loss = (
        masked_losses.sum()
        / valid_tokens
    )

    return (
        loss,
        masked_losses.sum(),
        valid_tokens,
    )


# ==========================================================
# Train one epoch
# ==========================================================

def train_one_epoch(
    model,
    loader,
    optimizer,
    scheduler=None,
    print_every=25,
):

    model.train()

    total_loss_sum = 0.0

    total_assistant_tokens = 0

    total_batches = len(loader)

    for batch_index, batch in enumerate(
        loader,
        start=1
    ):

        # --------------------------------------------------
        # Move tensors to device
        # --------------------------------------------------

        input_ids = batch[
            "input_ids"
        ].to(DEVICE)

        target_ids = batch[
            "target_ids"
        ].to(DEVICE)

        loss_mask = batch[
            "loss_mask"
        ].to(DEVICE)

        attention_mask = batch[
            "attention_mask"
        ].to(DEVICE)

        # --------------------------------------------------
        # Zero gradients
        # --------------------------------------------------

        optimizer.zero_grad(
            set_to_none=True
        )

        # --------------------------------------------------
        # Forward
        # --------------------------------------------------

        logits = model(
            input_ids,
            attention_mask=attention_mask,
        )

        # --------------------------------------------------
        # Loss
        # --------------------------------------------------

        (
            loss,
            batch_loss_sum,
            batch_valid_tokens,
        ) = calculate_sft_loss(
            logits=logits,
            target_ids=target_ids,
            loss_mask=loss_mask,
        )

        # --------------------------------------------------
        # Backpropagation
        # --------------------------------------------------

        loss.backward()

        # --------------------------------------------------
        # Gradient clipping
        # --------------------------------------------------

        gradient_norm = (
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                MAX_GRAD_NORM,
            )
        )

        # --------------------------------------------------
        # Optimizer
        # --------------------------------------------------

        optimizer.step()

        # --------------------------------------------------
        # Scheduler
        # --------------------------------------------------

        if scheduler is not None:
            scheduler.step()

        # --------------------------------------------------
        # Statistics
        # --------------------------------------------------

        total_loss_sum += (
            batch_loss_sum.item()
        )

        total_assistant_tokens += int(
            batch_valid_tokens.item()
        )

        # --------------------------------------------------
        # Progress
        # --------------------------------------------------

        if (
            batch_index == 1
            or batch_index % print_every == 0
            or batch_index == total_batches
        ):

            current_lr = (
                optimizer.param_groups[0]["lr"]
            )

            print(
                f"  Batch "
                f"{batch_index:>3}/"
                f"{total_batches:<3}"
                f" | loss={loss.item():.4f}"
                f" | grad={float(gradient_norm):.3f}"
                f" | lr={current_lr:.7f}"
            )

    if total_assistant_tokens == 0:

        raise RuntimeError(
            "No assistant tokens were found "
            "during training."
        )

    average_loss = (
        total_loss_sum
        / total_assistant_tokens
    )

    return (
        average_loss,
        total_assistant_tokens,
    )


# ==========================================================
# Validation
# ==========================================================

@torch.no_grad()
def validate(
    model,
    loader,
):

    model.eval()

    total_loss_sum = 0.0

    total_assistant_tokens = 0

    for batch in loader:

        # --------------------------------------------------
        # Move tensors to device
        # --------------------------------------------------

        input_ids = batch[
            "input_ids"
        ].to(DEVICE)

        target_ids = batch[
            "target_ids"
        ].to(DEVICE)

        loss_mask = batch[
            "loss_mask"
        ].to(DEVICE)

        attention_mask = batch[
            "attention_mask"
        ].to(DEVICE)

        # --------------------------------------------------
        # Forward
        # --------------------------------------------------

        logits = model(
            input_ids,
            attention_mask=attention_mask,
        )

        # --------------------------------------------------
        # Loss
        # --------------------------------------------------

        (
            loss,
            batch_loss_sum,
            batch_valid_tokens,
        ) = calculate_sft_loss(
            logits=logits,
            target_ids=target_ids,
            loss_mask=loss_mask,
        )

        # --------------------------------------------------
        # Statistics
        # --------------------------------------------------

        total_loss_sum += (
            batch_loss_sum.item()
        )

        total_assistant_tokens += int(
            batch_valid_tokens.item()
        )

    if total_assistant_tokens == 0:

        raise RuntimeError(
            "No assistant tokens were found "
            "during validation."
        )

    average_loss = (
        total_loss_sum
        / total_assistant_tokens
    )

    return (
        average_loss,
        total_assistant_tokens,
    )


# ==========================================================
# Checkpoint save
# ==========================================================

def save_checkpoint(
    model,
    optimizer,
    scheduler,
    epoch,
    train_loss,
    validation_loss,
    best_validation_loss,
    checkpoint_path,
    history,
    best_train_loss=None,
):

    CHECKPOINT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    config_dict = {
        "vocab_size":
            model.config.vocab_size,

        "max_seq_len":
            model.config.max_seq_len,

        "d_model":
            model.config.d_model,

        "num_heads":
            model.config.num_heads,

        "ffn_hidden_dim":
            model.config.ffn_hidden_dim,

        "num_layers":
            model.config.num_layers,

        "dropout":
            model.config.dropout,
    }

    checkpoint = {

        # --------------------------------------------------
        # Training state
        # --------------------------------------------------

        "epoch":
            epoch,

        "model_state_dict":
            model.state_dict(),

        "optimizer_state_dict":
            optimizer.state_dict(),

        "scheduler_state_dict":
            scheduler.state_dict()
            if scheduler is not None
            else None,

        # --------------------------------------------------
        # Metrics
        # --------------------------------------------------

        "train_loss":
            train_loss,

        "validation_loss":
            validation_loss,

        "best_validation_loss":
            best_validation_loss,

        "best_train_loss":
            best_train_loss,

        # --------------------------------------------------
        # Model configuration
        # --------------------------------------------------

        "config":
            config_dict,

        # --------------------------------------------------
        # Reproducibility
        # --------------------------------------------------

        "seed":
            SEED,

        # --------------------------------------------------
        # History
        # --------------------------------------------------

        "history":
            history,
    }

    torch.save(
        checkpoint,
        checkpoint_path,
    )


# ==========================================================
# Learning-rate scheduler
# ==========================================================

def build_scheduler(
    optimizer,
    total_steps,
    warmup_ratio,
):

    if total_steps <= 0:

        raise ValueError(
            "total_steps must be greater than zero."
        )

    warmup_steps = max(
        1,
        int(
            total_steps
            * warmup_ratio
        ),
    )

    def lr_lambda(step):

        # --------------------------------------------------
        # Warmup
        # --------------------------------------------------

        if step < warmup_steps:

            return (
                float(step + 1)
                / float(warmup_steps)
            )

        # --------------------------------------------------
        # Cosine decay
        # --------------------------------------------------

        remaining_steps = (
            total_steps
            - warmup_steps
        )

        if remaining_steps <= 0:

            return 1.0

        progress = (
            step
            - warmup_steps
        ) / remaining_steps

        progress = min(
            max(progress, 0.0),
            1.0,
        )

        return (
            0.5
            * (
                1.0
                + math.cos(
                    math.pi
                    * progress
                )
            )
        )

    scheduler = (
        torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda,
        )
    )

    return (
        scheduler,
        warmup_steps,
    )


# ==========================================================
# Print dataset statistics
# ==========================================================

def print_dataset_statistics(
    name,
    dataset,
):

    stats = dataset.statistics()

    print()
    print(name)

    print(
        f"  Records                : "
        f"{stats['records']}"
    )

    print(
        f"  Samples                : "
        f"{stats['samples']}"
    )

    print(
        f"  Total sequence tokens  : "
        f"{stats['total_sequence_tokens']}"
    )

    print(
        f"  Assistant loss tokens  : "
        f"{stats['assistant_loss_tokens']}"
    )

    print(
        f"  Assistant loss %       : "
        f"{stats['assistant_loss_percentage']:.2f}%"
    )

    print(
        f"  Padding tokens         : "
        f"{stats['padding_tokens']}"
    )

    print(
        f"  Padding %              : "
        f"{stats['padding_percentage']:.2f}%"
    )


# ==========================================================
# Build debug-overfit dataset
# ==========================================================

def build_debug_dataset(
    dataset,
    sample_count,
):

    if sample_count <= 0:

        raise ValueError(
            "Debug sample count must "
            "be greater than zero."
        )

    actual_count = min(
        sample_count,
        len(dataset),
    )

    # ------------------------------------------------------
    # Deterministic first N samples.
    # ------------------------------------------------------

    indices = list(
        range(actual_count)
    )

    debug_dataset = Subset(
        dataset,
        indices,
    )

    return debug_dataset


# ==========================================================
# Main
# ==========================================================

def main():

    args = parse_args()

    # ------------------------------------------------------
    # Seed
    # ------------------------------------------------------

    set_seed(SEED)

    debug_mode = args.debug_overfit

    # ======================================================
    # Effective configuration
    # ======================================================

    if debug_mode:

        effective_epochs = (
            args.debug_epochs
        )

        effective_batch_size = (
            DEBUG_OVERFIT_BATCH_SIZE
        )

        effective_learning_rate = (
            DEBUG_OVERFIT_LEARNING_RATE
        )

        effective_dropout = (
            DEBUG_OVERFIT_DROPOUT
        )

        effective_patience = (
            effective_epochs
        )

        effective_warmup_ratio = 0.0

        checkpoint_path = (
            DEBUG_CHECKPOINT
        )

    else:

        effective_epochs = EPOCHS

        effective_batch_size = BATCH_SIZE

        effective_learning_rate = (
            LEARNING_RATE
        )

        effective_dropout = DROPOUT

        effective_patience = (
            EARLY_STOPPING_PATIENCE
        )

        effective_warmup_ratio = (
            WARMUP_RATIO
        )

        checkpoint_path = (
            BEST_CHECKPOINT
        )

    # ======================================================
    # Header
    # ======================================================

    print()
    print("=" * 70)

    if debug_mode:

        print(
            "GENIEE V2 SFT - DEBUG OVERFIT MODE"
        )

    else:

        print(
            "GENIEE V2 SUPERVISED INSTRUCTION FINE-TUNING"
        )

    print("=" * 70)

    print()
    print(
        f"Device              : {DEVICE}"
    )

    print(
        f"Tokenizer            : "
        f"{TOKENIZER_PATH}"
    )

    print(
        f"Max sequence length : "
        f"{MAX_SEQ_LEN}"
    )

    print(
        f"Batch size           : "
        f"{effective_batch_size}"
    )

    print(
        f"Epochs               : "
        f"{effective_epochs}"
    )

    print(
        f"Learning rate        : "
        f"{effective_learning_rate}"
    )

    print(
        f"Weight decay         : "
        f"{WEIGHT_DECAY}"
    )

    print(
        f"Dropout              : "
        f"{effective_dropout}"
    )

    print(
        f"Checkpoint           : "
        f"{checkpoint_path}"
    )

    if debug_mode:

        print(
            f"Debug samples       : "
            f"{args.debug_samples}"
        )

        print(
            "Scheduler            : DISABLED"
        )

        print(
            "Best-model metric    : TRAIN LOSS"
        )

    else:

        print(
            f"Warmup ratio         : "
            f"{effective_warmup_ratio}"
        )

        print(
            f"Early stopping       : "
            f"{effective_patience}"
        )

        print(
            "Best-model metric    : VALIDATION LOSS"
        )

    # ======================================================
    # Tokenizer
    # ======================================================

    print()
    print(
        "Loading tokenizer..."
    )

    tokenizer = GenieeTokenizer(
        model_path=TOKENIZER_PATH
    )

    print(
        f"Vocabulary size      : "
        f"{tokenizer.vocab_size}"
    )

    print(
        f"PAD ID               : "
        f"{tokenizer.pad_id}"
    )

    print(
        f"BOS ID               : "
        f"{tokenizer.bos_id}"
    )

    print(
        f"EOS ID               : "
        f"{tokenizer.eos_id}"
    )

    # ======================================================
    # Datasets
    # ======================================================

    print()
    print(
        "Loading instruction datasets..."
    )

    full_train_dataset = (
        GenieeInstructionDataset(
            jsonl_path=(
                SPLIT_DIR
                / "train.jsonl"
            ),

            tokenizer=tokenizer,

            max_seq_len=MAX_SEQ_LEN,

            pad_id=tokenizer.pad_id,
        )
    )

    full_validation_dataset = (
        GenieeInstructionDataset(
            jsonl_path=(
                SPLIT_DIR
                / "validation.jsonl"
            ),

            tokenizer=tokenizer,

            max_seq_len=MAX_SEQ_LEN,

            pad_id=tokenizer.pad_id,
        )
    )

    print(
        f"Full train samples       : "
        f"{len(full_train_dataset)}"
    )

    print(
        f"Validation samples       : "
        f"{len(full_validation_dataset)}"
    )

    # ======================================================
    # Debug dataset selection
    # ======================================================

    if debug_mode:

        train_dataset = build_debug_dataset(
            full_train_dataset,
            args.debug_samples,
        )

        print()
        print(
            "DEBUG DATASET"
        )

        print(
            f"  Requested samples      : "
            f"{args.debug_samples}"
        )

        print(
            f"  Actual samples         : "
            f"{len(train_dataset)}"
        )

        print()
        print(
            "IMPORTANT:"
        )

        print(
            "  This run is NOT normal training."
        )

        print(
            "  The purpose is to verify that "
            "Geniee can memorize a tiny dataset."
        )

        print(
            "  Very low training loss is expected."
        )

        print()
        print(
            "  Debug checkpoint will be saved to:"
        )

        print(
            f"  {DEBUG_CHECKPOINT}"
        )

    else:

        train_dataset = full_train_dataset

    # ======================================================
    # Dataset statistics
    # ======================================================

    print_dataset_statistics(
        "TRAIN DATASET",
        full_train_dataset,
    )

    print_dataset_statistics(
        "VALIDATION DATASET",
        full_validation_dataset,
    )

    # ======================================================
    # DataLoaders
    # ======================================================

    train_loader = DataLoader(
        train_dataset,

        batch_size=effective_batch_size,

        shuffle=True,

        num_workers=NUM_WORKERS,
    )

    validation_loader = DataLoader(
        full_validation_dataset,

        batch_size=effective_batch_size,

        shuffle=False,

        num_workers=NUM_WORKERS,
    )

    # ======================================================
    # Model
    # ======================================================

    print()
    print(
        "Creating Geniee model..."
    )

    model = build_model(
        tokenizer=tokenizer,
        dropout=effective_dropout,
    )

    model = model.to(
        DEVICE
    )

    # ------------------------------------------------------
    # Verify model vocabulary
    # ------------------------------------------------------

    if (
        model.config.vocab_size
        != tokenizer.vocab_size
    ):

        raise RuntimeError(
            "Vocabulary mismatch.\n"
            f"Tokenizer vocab size: "
            f"{tokenizer.vocab_size}\n"
            f"Model vocab size: "
            f"{model.config.vocab_size}"
        )

    parameter_count = count_parameters(
        model
    )

    print(
        f"Parameters           : "
        f"{parameter_count:,}"
    )

    print(
        f"Parameters (M)       : "
        f"{parameter_count / 1_000_000:.2f}M"
    )

    print()
    print(
        "Model configuration:"
    )

    print(
        f"  vocab_size         : "
        f"{model.config.vocab_size}"
    )

    print(
        f"  max_seq_len        : "
        f"{model.config.max_seq_len}"
    )

    print(
        f"  d_model            : "
        f"{model.config.d_model}"
    )

    print(
        f"  num_heads          : "
        f"{model.config.num_heads}"
    )

    print(
        f"  head_dim           : "
        f"{model.config.head_dim}"
    )

    print(
        f"  ffn_hidden_dim     : "
        f"{model.config.ffn_hidden_dim}"
    )

    print(
        f"  num_layers         : "
        f"{model.config.num_layers}"
    )

    print(
        f"  dropout            : "
        f"{model.config.dropout}"
    )

    # ======================================================
    # Optimizer
    # ======================================================

    optimizer = torch.optim.AdamW(
        model.parameters(),

        lr=effective_learning_rate,

        weight_decay=WEIGHT_DECAY,
    )

    # ======================================================
    # Scheduler
    # ======================================================

    scheduler = None

    if debug_mode:

        total_steps = (
            len(train_loader)
            * effective_epochs
        )

        warmup_steps = 0

        print()
        print(
            "Scheduler disabled for "
            "debug-overfit."
        )

    else:

        total_steps = (
            len(train_loader)
            * effective_epochs
        )

        (
            scheduler,
            warmup_steps,
        ) = build_scheduler(
            optimizer=optimizer,

            total_steps=total_steps,

            warmup_ratio=effective_warmup_ratio,
        )

        print()
        print(
            f"Total training steps : "
            f"{total_steps}"
        )

        print(
            f"Warmup steps         : "
            f"{warmup_steps}"
        )

    # ======================================================
    # Training state
    # ======================================================

    # ------------------------------------------------------
    # Normal mode:
    #     best validation loss
    #
    # Debug mode:
    #     best training loss
    # ------------------------------------------------------

    best_validation_loss = float(
        "inf"
    )

    best_train_loss = float(
        "inf"
    )

    best_epoch = 0

    epochs_without_improvement = 0

    history = []

    # ======================================================
    # Training
    # ======================================================

    print()
    print("=" * 70)

    if debug_mode:

        print(
            "STARTING DEBUG OVERFIT TEST"
        )

    else:

        print(
            "STARTING SFT TRAINING"
        )

    print("=" * 70)

    for epoch in range(
        1,
        effective_epochs + 1,
    ):

        print()
        print(
            "-" * 70
        )

        print(
            f"Epoch "
            f"{epoch}/{effective_epochs}"
        )

        print(
            "-" * 70
        )

        # --------------------------------------------------
        # Training
        # --------------------------------------------------

        train_loss, train_tokens = (
            train_one_epoch(
                model=model,

                loader=train_loader,

                optimizer=optimizer,

                scheduler=scheduler,

                print_every=(
                    DEBUG_PRINT_EVERY
                    if debug_mode
                    else 25
                ),
            )
        )

        # --------------------------------------------------
        # Validation
        # --------------------------------------------------

        validation_loss, validation_tokens = (
            validate(
                model=model,

                loader=validation_loader,
            )
        )

        current_lr = (
            optimizer.param_groups[0]["lr"]
        )

        print()
        print(
            f"Epoch {epoch}/{effective_epochs}"
            f" | train_loss={train_loss:.4f}"
            f" | val_loss={validation_loss:.4f}"
            f" | lr={current_lr:.7f}"
        )

        print(
            f"  Train assistant tokens : "
            f"{train_tokens}"
        )

        print(
            f"  Validation assistant tokens : "
            f"{validation_tokens}"
        )

        # ==================================================
        # Debug-overfit progress
        # ==================================================

        if debug_mode:

            print()

            print(
                f"  DEBUG OVERFIT LOSS: "
                f"{train_loss:.6f}"
            )

            if train_loss < 1.0:

                print(
                    "  ✓ Loss is below 1.0"
                )

            if train_loss < 0.5:

                print(
                    "  ✓ Loss is below 0.5"
                )

            if train_loss < 0.1:

                print(
                    "  ✓ Excellent memorization signal"
                )

        # ==================================================
        # History
        # ==================================================

        history.append(
            {
                "epoch":
                    epoch,

                "train_loss":
                    train_loss,

                "validation_loss":
                    validation_loss,

                "learning_rate":
                    current_lr,
            }
        )

        # ==================================================
        # BEST MODEL LOGIC
        # ==================================================

        if debug_mode:

            # ------------------------------------------------
            # DEBUG MODE:
            #
            # Save model with LOWEST TRAIN LOSS.
            #
            # Validation is intentionally ignored when
            # selecting the debug checkpoint.
            # ------------------------------------------------

            if train_loss < best_train_loss:

                improvement = (
                    best_train_loss
                    - train_loss
                )

                best_train_loss = train_loss

                best_epoch = epoch

                epochs_without_improvement = 0

                save_checkpoint(
                    model=model,

                    optimizer=optimizer,

                    scheduler=scheduler,

                    epoch=epoch,

                    train_loss=train_loss,

                    validation_loss=validation_loss,

                    best_validation_loss=(
                        best_validation_loss
                    ),

                    checkpoint_path=(
                        checkpoint_path
                    ),

                    history=history,

                    best_train_loss=(
                        best_train_loss
                    ),
                )

                print()
                print(
                    "  ✓ New best DEBUG model"
                )

                if math.isfinite(improvement):

                    print(
                        f"  Train loss improvement : "
                        f"{improvement:.6f}"
                    )

                print(
                    f"  Best train loss        : "
                    f"{best_train_loss:.6f}"
                )

                print(
                    f"  Saved                  : "
                    f"{checkpoint_path}"
                )

            else:

                epochs_without_improvement += 1

                print()
                print(
                    "  No training-loss improvement."
                )

                print(
                    f"  Debug progress: "
                    f"{epochs_without_improvement}/"
                    f"{effective_patience}"
                )

        else:

            # ------------------------------------------------
            # NORMAL MODE:
            #
            # Save model with LOWEST VALIDATION LOSS.
            # ------------------------------------------------

            if (
                validation_loss
                < best_validation_loss
            ):

                improvement = (
                    best_validation_loss
                    - validation_loss
                )

                best_validation_loss = (
                    validation_loss
                )

                best_epoch = epoch

                epochs_without_improvement = 0

                save_checkpoint(
                    model=model,

                    optimizer=optimizer,

                    scheduler=scheduler,

                    epoch=epoch,

                    train_loss=train_loss,

                    validation_loss=validation_loss,

                    best_validation_loss=(
                        best_validation_loss
                    ),

                    checkpoint_path=(
                        checkpoint_path
                    ),

                    history=history,

                    best_train_loss=(
                        best_train_loss
                    ),
                )

                print()
                print(
                    "  ✓ New best model"
                )

                if math.isfinite(improvement):

                    print(
                        f"  Validation improvement : "
                        f"{improvement:.6f}"
                    )

                print(
                    f"  Saved : "
                    f"{checkpoint_path}"
                )

            else:

                epochs_without_improvement += 1

                print()
                print(
                    "  No validation improvement."
                )

                print(
                    f"  Patience: "
                    f"{epochs_without_improvement}/"
                    f"{effective_patience}"
                )

                # ------------------------------------------------
                # Normal early stopping
                # ------------------------------------------------

                if (
                    epochs_without_improvement
                    >= effective_patience
                ):

                    print()
                    print(
                        "Early stopping triggered."
                    )

                    break

    # ======================================================
    # Final summary
    # ======================================================

    print()
    print("=" * 70)

    if debug_mode:

        print(
            "GENIEE DEBUG OVERFIT TEST COMPLETE"
        )

    else:

        print(
            "SFT TRAINING COMPLETE"
        )

    print("=" * 70)

    # ------------------------------------------------------
    # Common final values
    # ------------------------------------------------------

    final_train_loss = (
        history[-1]["train_loss"]
        if history
        else float("inf")
    )

    final_validation_loss = (
        history[-1]["validation_loss"]
        if history
        else float("inf")
    )

    # ------------------------------------------------------
    # Summary
    # ------------------------------------------------------

    print(
        f"Best epoch          : "
        f"{best_epoch}"
    )

    if debug_mode:

        print(
            f"Best train loss     : "
            f"{best_train_loss:.6f}"
        )

        print(
            f"Final train loss    : "
            f"{final_train_loss:.6f}"
        )

        print(
            f"Final validation    : "
            f"{final_validation_loss:.6f}"
        )

    else:

        print(
            f"Best validation loss: "
            f"{best_validation_loss:.6f}"
        )

        print(
            f"Final train loss    : "
            f"{final_train_loss:.6f}"
        )

        print(
            f"Final validation    : "
            f"{final_validation_loss:.6f}"
        )

    print(
        f"Checkpoint          : "
        f"{checkpoint_path}"
    )

    # ======================================================
    # Debug interpretation
    # ======================================================

    if debug_mode:

        print()
        print(
            "DEBUG INTERPRETATION"
        )

        if best_train_loss < 0.1:

            print(
                "  ✓ Excellent."
            )

            print(
                "  Geniee successfully memorized "
                "the tiny training subset."
            )

            print()
            print(
                "  The Transformer training pipeline "
                "is functioning correctly."
            )

        elif best_train_loss < 0.5:

            print(
                "  ✓ Good."
            )

            print(
                "  Model is strongly fitting "
                "the tiny training subset."
            )

        elif best_train_loss < 1.0:

            print(
                "  ✓ Promising."
            )

            print(
                "  Model is learning the subset, "
                "but has not fully memorized it."
            )

        elif best_train_loss < 2.0:

            print(
                "  ⚠ Partial memorization."
            )

            print(
                "  Investigate masking, "
                "attention, optimizer, "
                "or learning rate."
            )

        else:

            print(
                "  ✗ Overfit test failed."
            )

            print(
                "  Do not scale training yet."
            )

            print(
                "  Inspect the dataset, "
                "loss mask, target shift, "
                "causal attention, and model."
            )

    # ======================================================
    # Training history
    # ======================================================

    print()
    print(
        "Training history:"
    )

    for item in history:

        print(
            f"  Epoch "
            f"{item['epoch']:>3}"
            f" | train="
            f"{item['train_loss']:.6f}"
            f" | val="
            f"{item['validation_loss']:.6f}"
            f" | lr="
            f"{item['learning_rate']:.7f}"
        )

    print()
    print("=" * 70)


# ==========================================================
# Entry point
# ==========================================================

if __name__ == "__main__":

    main()