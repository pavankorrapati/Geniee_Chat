from pathlib import Path

import torch

from data.build_dataset import GenieeDataModule


def get_data_module():

    project_root = (
        Path(__file__).resolve().parent.parent
    )

    corpus_dir = (
        project_root / "corpus"
    )

    tokenizer_path = (
        project_root
        / "tokenizer"
        / "artifacts"
        / "geniee.model"
    )

    return GenieeDataModule(
        corpus_dir=corpus_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=8,
        batch_size=2,
        stride=4,
        validation_split=0.1,
        shuffle_train=False
    )


def test_data_module_builds_datasets():

    data_module = get_data_module()

    train_dataset, validation_dataset = (
        data_module.build_datasets()
    )

    print()
    print(
        "Training samples:",
        len(train_dataset)
    )

    print(
        "Validation samples:",
        len(validation_dataset)
    )

    assert len(train_dataset) > 0

    assert len(validation_dataset) > 0


def test_data_module_builds_dataloaders():

    data_module = get_data_module()

    train_loader, validation_loader = (
        data_module.build()
    )

    # --------------------------------------------------
    # Get training batch
    # --------------------------------------------------

    train_input, train_target = next(
        iter(train_loader)
    )

    # --------------------------------------------------
    # Get validation batch
    # --------------------------------------------------

    validation_input, validation_target = next(
        iter(validation_loader)
    )

    print()
    print(
        "Training input shape:",
        train_input.shape
    )

    print(
        "Training target shape:",
        train_target.shape
    )

    print(
        "Validation input shape:",
        validation_input.shape
    )

    print(
        "Validation target shape:",
        validation_target.shape
    )

    # --------------------------------------------------
    # Validate training batch
    # --------------------------------------------------

    assert train_input.ndim == 2

    assert train_target.ndim == 2

    assert train_input.shape == train_target.shape

    assert train_input.shape[1] == 8

    # --------------------------------------------------
    # Validate validation batch
    # --------------------------------------------------

    assert validation_input.ndim == 2

    assert validation_target.ndim == 2

    assert (
        validation_input.shape
        == validation_target.shape
    )

    assert validation_input.shape[1] == 8

    # --------------------------------------------------
    # Validate data type
    # --------------------------------------------------

    assert train_input.dtype == torch.long

    assert train_target.dtype == torch.long


def test_data_module_summary():

    data_module = get_data_module()

    data_module.build()

    data_module.summary()

    assert data_module.train_loader is not None

    assert (
        data_module.validation_loader
        is not None
    )