from __future__ import annotations

from pathlib import Path
import random
import sys
import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# PROJECT IMPORTS
# ============================================================

from model.config import GenieeConfig
from model.geniee_model import GenieeModel
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

CHECKPOINT_DIR = (
    PROJECT_ROOT
    / "checkpoints"
)

BEST_CHECKPOINT = (
    CHECKPOINT_DIR
    / "geniee_pretrain_best.pt"
)


# ============================================================
# MODEL CONFIGURATION
# ============================================================

MAX_SEQ_LEN = 256
D_MODEL = 512
NUM_HEADS = 8
FFN_HIDDEN_DIM = 2048
NUM_LAYERS = 8
DROPOUT = 0.1


# ============================================================
# TRAINING CONFIGURATION
# ============================================================

EPOCHS = 10

BATCH_SIZE = 2

LEARNING_RATE = 1e-4

WEIGHT_DECAY = 0.01

GRAD_CLIP = 1.0

STRIDE = 64

NUM_WORKERS = 0

SEED = 42


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# RANDOM SEED
# ============================================================

def set_seed(seed: int):

    random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# PARAMETER COUNT
# ============================================================

def count_parameters(model):

    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )


# ============================================================
# LOSS
# ============================================================

def calculate_loss(
    logits: torch.Tensor,
    target_ids: torch.Tensor,
) -> torch.Tensor:

    batch_size = logits.size(0)
    sequence_length = logits.size(1)
    vocab_size = logits.size(2)

    logits = logits.reshape(
        batch_size * sequence_length,
        vocab_size,
    )

    targets = target_ids.reshape(
        batch_size * sequence_length
    )

    loss = nn.functional.cross_entropy(
        logits,
        targets,
    )

    return loss


# ============================================================
# TRAIN ONE EPOCH
# ============================================================

def train_one_epoch(
    model,
    dataloader,
    optimizer,
):

    model.train()

    total_loss = 0.0

    total_batches = 0

    for batch in dataloader:

        input_ids = batch["input_ids"].to(
            DEVICE
        )

        target_ids = batch["target_ids"].to(
            DEVICE
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        logits = model(
            input_ids
        )

        loss = calculate_loss(
            logits,
            target_ids,
        )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            GRAD_CLIP,
        )

        optimizer.step()

        total_loss += loss.item()

        total_batches += 1

    if total_batches == 0:
        return float("nan")

    return total_loss / total_batches


# ============================================================
# VALIDATION
# ============================================================

@torch.no_grad()
def validate(
    model,
    dataloader,
):

    model.eval()

    total_loss = 0.0

    total_batches = 0

    for batch in dataloader:

        input_ids = batch["input_ids"].to(
            DEVICE
        )

        target_ids = batch["target_ids"].to(
            DEVICE
        )

        logits = model(
            input_ids
        )

        loss = calculate_loss(
            logits,
            target_ids,
        )

        total_loss += loss.item()

        total_batches += 1

    if total_batches == 0:
        return float("nan")

    return total_loss / total_batches


# ============================================================
# CHECKPOINT
# ============================================================

def save_checkpoint(
    model,
    optimizer,
    epoch,
    train_loss,
    validation_loss,
    checkpoint_path,
):

    CHECKPOINT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "train_loss": train_loss,
        "validation_loss": validation_loss,

        "config": {
            "vocab_size": model.config.vocab_size,
            "max_seq_len": model.config.max_seq_len,
            "d_model": model.config.d_model,
            "num_heads": model.config.num_heads,
            "ffn_hidden_dim": model.config.ffn_hidden_dim,
            "num_layers": model.config.num_layers,
            "dropout": model.config.dropout,
        },

        "seed": SEED,
    }

    torch.save(
        checkpoint,
        checkpoint_path,
    )


# ============================================================
# MAIN
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description="Pretrain or continue pretraining Geniee."
    )

    parser.add_argument(
        "--train-file",
        type=Path,
        default=TRAIN_PATH,
        help="Training text split.",
    )

    parser.add_argument(
        "--validation-file",
        type=Path,
        default=VALIDATION_PATH,
        help="Validation text split.",
    )

    parser.add_argument(
        "--init-checkpoint",
        type=Path,
        default=None,
        help="Optional checkpoint from which to continue training.",
    )

    parser.add_argument(
        "--output-checkpoint",
        type=Path,
        default=BEST_CHECKPOINT,
        help="Path for the best checkpoint produced by this run.",
    )

    return parser.parse_args()

def main():

    args = parse_args()

    print()
    print("=" * 70)
    print("GENIEE BASE PRETRAINING")
    print("=" * 70)

    print()
    print(f"Device     : {DEVICE}")
    print(f"Tokenizer  : {TOKENIZER_PATH}")
    print(f"Train data : {args.train_file}")
    print(f"Val data   : {args.validation_file}")
    print(f"Init model : {args.init_checkpoint or 'random weights'}")
    print(f"Output     : {args.output_checkpoint}")

    print()
    print("-" * 70)
    print("TRAINING CONFIGURATION")
    print("-" * 70)

    print()
    print(f"Epochs          : {EPOCHS}")
    print(f"Batch size      : {BATCH_SIZE}")
    print(f"Learning rate   : {LEARNING_RATE}")
    print(f"Weight decay    : {WEIGHT_DECAY}")
    print(f"Max seq length  : {MAX_SEQ_LEN}")
    print(f"Gradient clip   : {GRAD_CLIP}")

    # --------------------------------------------------------
    # Seed
    # --------------------------------------------------------

    set_seed(SEED)

    # --------------------------------------------------------
    # Tokenizer
    # --------------------------------------------------------

    tokenizer = GenieeTokenizer(
        model_path=TOKENIZER_PATH
    )

    print()
    print(f"Vocabulary size : {tokenizer.vocab_size}")

    # --------------------------------------------------------
    # Configuration
    # --------------------------------------------------------

    config = GenieeConfig(
        vocab_size=tokenizer.vocab_size,
        max_seq_len=MAX_SEQ_LEN,
        d_model=D_MODEL,
        num_heads=NUM_HEADS,
        ffn_hidden_dim=FFN_HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
    )

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    print()
    print("Loading datasets...")

    train_dataset = PretrainingDataset(
        text_path=args.train_file,
        tokenizer=tokenizer,
        max_seq_len=MAX_SEQ_LEN,
    )

    validation_dataset = PretrainingDataset(
        text_path=args.validation_file,
        tokenizer=tokenizer,
        max_seq_len=MAX_SEQ_LEN,
    )

    print()
    print(
        f"Train tokens       : "
        f"{train_dataset.total_tokens:,}"
    )

    print(
        f"Train sequences    : "
        f"{train_dataset.num_sequences:,}"
    )

    print(
        f"Validation tokens  : "
        f"{validation_dataset.total_tokens:,}"
    )

    print(
        f"Validation sequences: "
        f"{validation_dataset.num_sequences:,}"
    )

    if len(train_dataset) == 0:
        raise RuntimeError(
            "Training dataset contains zero complete "
            "sequences. Increase the corpus size."
        )

    if len(validation_dataset) == 0:
        raise RuntimeError(
            "Validation dataset contains zero complete "
            "sequences. Increase the validation corpus size."
        )

    # --------------------------------------------------------
    # DataLoaders
    # --------------------------------------------------------

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    print()
    print("Building Geniee model...")

    model = GenieeModel(
        config=config
    ).to(DEVICE)

    if args.init_checkpoint is not None:

        if not args.init_checkpoint.exists():
            raise FileNotFoundError(
                f"Initial checkpoint not found: "
                f"{args.init_checkpoint}"
            )

        print()
        print(f"Loading initial checkpoint: {args.init_checkpoint}")

        checkpoint = torch.load(
            args.init_checkpoint,
            map_location=DEVICE,
        )

        model.load_state_dict(
            checkpoint["model_state_dict"]
        )

    print(
        f"Parameters : "
        f"{count_parameters(model):,}"
    )

    # --------------------------------------------------------
    # Optimizer
    # --------------------------------------------------------

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    best_validation_loss = float("inf")

    print()
    print("=" * 70)
    print("TRAINING")
    print("=" * 70)

    for epoch in range(
        1,
        EPOCHS + 1,
    ):

        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
        )

        validation_loss = validate(
            model,
            validation_loader,
        )

        print(
            f"Epoch {epoch:03d}/{EPOCHS:03d} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Validation Loss: {validation_loss:.4f}"
        )

        # ----------------------------------------------------
        # Save best checkpoint
        # ----------------------------------------------------

        if validation_loss < best_validation_loss:

            best_validation_loss = validation_loss

            save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                train_loss=train_loss,
                validation_loss=validation_loss,
                checkpoint_path=args.output_checkpoint,
            )

            print(
                f"  ✓ Best checkpoint saved: "
                f"{BEST_CHECKPOINT}"
            )

    print()
    print("=" * 70)
    print("PRETRAINING COMPLETE")
    print("=" * 70)

    print()
    print(
        f"Best validation loss : "
        f"{best_validation_loss:.4f}"
    )

    print(
        f"Checkpoint           : "
        f"{args.output_checkpoint}"
    )

    print()


if __name__ == "__main__":
    main()