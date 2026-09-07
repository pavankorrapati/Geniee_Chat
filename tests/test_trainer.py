from pathlib import Path

import torch

from data.build_dataset import GenieeDataModule
from model.config import GenieeConfig
from model.geniee_model import GenieeModel
from training.trainer import GenieeTrainer


# ==========================================================
# Helper
# ==========================================================

def create_test_components():

    project_root = (
        Path(__file__).resolve().parent.parent
    )

    # --------------------------------------------------
    # Data paths
    # --------------------------------------------------

    corpus_dir = (
        project_root / "corpus"
    )

    tokenizer_path = (
        project_root
        / "tokenizer"
        / "artifacts"
        / "geniee.model"
    )

    # --------------------------------------------------
    # Data module
    # --------------------------------------------------

    data_module = GenieeDataModule(

        corpus_dir=corpus_dir,

        tokenizer_path=tokenizer_path,

        max_seq_len=8,

        batch_size=2,

        stride=4,

        validation_split=0.1,

        shuffle_train=False
    )

    train_loader, validation_loader = (
        data_module.build()
    )

    # --------------------------------------------------
    # Geniee configuration
    #
    # IMPORTANT:
    # Use the existing GenieeConfig defaults.
    # We only override values that are known to
    # be required for this test.
    # --------------------------------------------------

    config = GenieeConfig(
        vocab_size=256,
        max_seq_len=8
    )

    # --------------------------------------------------
    # Model
    # --------------------------------------------------

    model = GenieeModel(
        config
    )

    # --------------------------------------------------
    # Trainer
    # --------------------------------------------------

    checkpoint_dir = (
        project_root
        / "test_checkpoints"
    )

    trainer = GenieeTrainer(

        model=model,

        train_loader=train_loader,

        validation_loader=validation_loader,

        learning_rate=3e-4,

        weight_decay=0.01,

        max_grad_norm=1.0,

        device="cpu",

        checkpoint_dir=checkpoint_dir
    )

    return trainer


# ==========================================================
# Test 1
# ==========================================================

def test_trainer_train_step():

    trainer = create_test_components()

    # --------------------------------------------------
    # Get one batch
    # --------------------------------------------------

    input_ids, target_ids = next(
        iter(trainer.train_loader)
    )

    # --------------------------------------------------
    # Training step
    # --------------------------------------------------

    loss = trainer.train_step(
        input_ids,
        target_ids
    )

    print()
    print(
        "Training loss:",
        loss
    )

    # --------------------------------------------------
    # Validate loss
    # --------------------------------------------------

    assert isinstance(
        loss,
        float
    )

    assert loss > 0

    # --------------------------------------------------
    # One optimizer step should have happened
    # --------------------------------------------------

    assert trainer.global_step == 1


# ==========================================================
# Test 2
# ==========================================================

def test_trainer_validation():

    trainer = create_test_components()

    # --------------------------------------------------
    # Validation
    # --------------------------------------------------

    validation_loss = (
        trainer.validate()
    )

    print()
    print(
        "Validation loss:",
        validation_loss
    )

    # --------------------------------------------------
    # Validate
    # --------------------------------------------------

    assert isinstance(
        validation_loss,
        float
    )

    assert validation_loss > 0

    assert len(
        trainer.validation_losses
    ) == 1


# ==========================================================
# Test 3
# ==========================================================

def test_trainer_single_epoch():

    trainer = create_test_components()

    # --------------------------------------------------
    # Train one epoch
    # --------------------------------------------------

    train_loss = (
        trainer.train_epoch()
    )

    print()
    print(
        "Epoch training loss:",
        train_loss
    )

    # --------------------------------------------------
    # Validate
    # --------------------------------------------------

    assert isinstance(
        train_loss,
        float
    )

    assert train_loss > 0

    # --------------------------------------------------
    # train_epoch() itself does not modify
    # current_epoch.
    #
    # current_epoch is controlled by train().
    # --------------------------------------------------

    assert trainer.current_epoch == 0

    # --------------------------------------------------
    # One loss should have been recorded
    # --------------------------------------------------

    assert len(
        trainer.train_losses
    ) == 1


# ==========================================================
# Test 4
# ==========================================================

def test_trainer_checkpoint():

    trainer = create_test_components()

    # --------------------------------------------------
    # Train one epoch
    # --------------------------------------------------

    trainer.train_epoch()

    # --------------------------------------------------
    # Save checkpoint
    # --------------------------------------------------

    checkpoint_path = (
        trainer.save_checkpoint(
            epoch=1
        )
    )

    print()
    print(
        "Checkpoint:",
        checkpoint_path
    )

    # --------------------------------------------------
    # Check file
    # --------------------------------------------------

    assert checkpoint_path.exists()

    # --------------------------------------------------
    # Create a second trainer
    # --------------------------------------------------

    trainer2 = (
        create_test_components()
    )

    # --------------------------------------------------
    # Load checkpoint
    # --------------------------------------------------

    trainer2.load_checkpoint(
        checkpoint_path
    )

    # --------------------------------------------------
    # Validate epoch
    # --------------------------------------------------

    assert (
        trainer2.current_epoch
        == 1
    )

    # --------------------------------------------------
    # Validate global step
    # --------------------------------------------------

    assert (
        trainer2.global_step
        == trainer.global_step
    )

    # --------------------------------------------------
    # Validate training history
    # --------------------------------------------------

    assert len(
        trainer2.train_losses
    ) == len(
        trainer.train_losses
    )

    assert len(
        trainer2.validation_losses
    ) == len(
        trainer.validation_losses
    )


# ==========================================================
# Test 5
# ==========================================================

def test_training_loss_can_decrease():

    trainer = create_test_components()

    losses = []

    # --------------------------------------------------
    # Train several epochs
    # --------------------------------------------------

    for _ in range(5):

        loss = trainer.train_epoch()

        losses.append(
            loss
        )

    # --------------------------------------------------
    # Display
    # --------------------------------------------------

    print()
    print(
        "Training losses:"
    )

    for index, loss in enumerate(
        losses,
        start=1
    ):

        print(
            f"Epoch {index}: "
            f"{loss:.6f}"
        )

    # --------------------------------------------------
    # Every loss should be positive
    # --------------------------------------------------

    assert all(
        loss > 0
        for loss in losses
    )

    # --------------------------------------------------
    # We expect the model to learn from
    # the tiny training corpus.
    # --------------------------------------------------

    assert (
        losses[-1]
        < losses[0]
    )