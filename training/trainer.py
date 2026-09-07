from pathlib import Path

import torch
import torch.nn as nn


class GenieeTrainer:
    """
    Training engine for the Geniee language model.

    Responsibilities:
        - Model training
        - Validation
        - Optimizer updates
        - Gradient clipping
        - Checkpoint saving
        - Checkpoint loading
        - Loss tracking
    """

    def __init__(
        self,
        model,
        train_loader,
        validation_loader,
        learning_rate=3e-4,
        weight_decay=0.01,
        max_grad_norm=1.0,
        device=None,
        checkpoint_dir="checkpoints"
    ):

        # ==================================================
        # Device
        # ==================================================

        if device is None:

            self.device = torch.device(
                "cuda"
                if torch.cuda.is_available()
                else "cpu"
            )

        else:

            self.device = torch.device(
                device
            )

        # ==================================================
        # Model
        # ==================================================

        self.model = model.to(
            self.device
        )

        # ==================================================
        # Data
        # ==================================================

        self.train_loader = train_loader

        self.validation_loader = (
            validation_loader
        )

        # ==================================================
        # Training configuration
        # ==================================================

        self.learning_rate = learning_rate

        self.weight_decay = weight_decay

        self.max_grad_norm = max_grad_norm

        # ==================================================
        # Loss function
        # ==================================================

        # self.criterion = nn.CrossEntropyLoss()
        self.criterion = nn.CrossEntropyLoss(ignore_index=0)

        # ==================================================
        # Optimizer
        # ==================================================

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )

        # ==================================================
        # Checkpoints
        # ==================================================

        self.checkpoint_dir = Path(
            checkpoint_dir
        )

        self.checkpoint_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        # ==================================================
        # Training state
        # ==================================================

        self.current_epoch = 0

        self.global_step = 0

        self.train_losses = []

        self.validation_losses = []
        self.best_validation_loss = float("inf")
        self.best_epoch = 0

    # ======================================================
    # Single training batch
    # ======================================================

    def train_step(
        self,
        input_ids,
        target_ids
    ):

        # --------------------------------------------------
        # Training mode
        # --------------------------------------------------

        self.model.train()

        # --------------------------------------------------
        # Move tensors to device
        # --------------------------------------------------

        input_ids = input_ids.to(
            self.device
        )

        target_ids = target_ids.to(
            self.device
        )

        # --------------------------------------------------
        # Clear previous gradients
        # --------------------------------------------------

        self.optimizer.zero_grad(
            set_to_none=True
        )

        # --------------------------------------------------
        # Forward pass
        # --------------------------------------------------

        logits = self.model(
            input_ids
        )

        # --------------------------------------------------
        # Reshape logits
        #
        # logits:
        # [batch, sequence, vocab]
        #
        # CrossEntropyLoss expects:
        # [batch * sequence, vocab]
        # --------------------------------------------------

        batch_size = logits.size(0)

        sequence_length = logits.size(1)

        vocabulary_size = logits.size(2)

        logits = logits.reshape(
            batch_size * sequence_length,
            vocabulary_size
        )

        target_ids = target_ids.reshape(
            batch_size * sequence_length
        )

        # --------------------------------------------------
        # Calculate loss
        # --------------------------------------------------

        loss = self.criterion(
            logits,
            target_ids
        )

        # --------------------------------------------------
        # Backpropagation
        # --------------------------------------------------

        loss.backward()

        # --------------------------------------------------
        # Gradient clipping
        # --------------------------------------------------

        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            self.max_grad_norm
        )

        # --------------------------------------------------
        # Update model
        # --------------------------------------------------

        self.optimizer.step()

        # --------------------------------------------------
        # Global step
        # --------------------------------------------------

        self.global_step += 1

        return loss.item()

    # ======================================================
    # Train one epoch
    # ======================================================

    def train_epoch(self):

        self.model.train()

        total_loss = 0.0

        number_of_batches = 0

        for input_ids, target_ids in (
            self.train_loader
        ):

            loss = self.train_step(
                input_ids,
                target_ids
            )

            total_loss += loss

            number_of_batches += 1

        if number_of_batches == 0:

            raise RuntimeError(
                "Training DataLoader is empty."
            )

        average_loss = (
            total_loss
            / number_of_batches
        )

        self.train_losses.append(
            average_loss
        )

        return average_loss

    # ======================================================
    # Validation
    # ======================================================

    @torch.no_grad()
    def validate(self):

        self.model.eval()

        total_loss = 0.0

        number_of_batches = 0

        for input_ids, target_ids in (
            self.validation_loader
        ):

            input_ids = input_ids.to(
                self.device
            )

            target_ids = target_ids.to(
                self.device
            )

            # ----------------------------------------------
            # Forward pass
            # ----------------------------------------------

            logits = self.model(
                input_ids
            )

            # ----------------------------------------------
            # Reshape
            # ----------------------------------------------

            batch_size = logits.size(0)

            sequence_length = logits.size(1)

            vocabulary_size = logits.size(2)

            logits = logits.reshape(
                batch_size * sequence_length,
                vocabulary_size
            )

            target_ids = target_ids.reshape(
                batch_size * sequence_length
            )

            # ----------------------------------------------
            # Loss
            # ----------------------------------------------

            loss = self.criterion(
                logits,
                target_ids
            )

            total_loss += loss.item()

            number_of_batches += 1

        if number_of_batches == 0:

            raise RuntimeError(
                "Validation DataLoader is empty."
            )

        average_loss = (
            total_loss
            / number_of_batches
        )

        self.validation_losses.append(
            average_loss
        )

        return average_loss

    # ======================================================
    # Train
    # ======================================================

    # def train(
    #     self,
    #     epochs,
    #     save_every=1
    # ):

    #     if epochs <= 0:

    #         raise ValueError(
    #             "epochs must be greater than zero."
    #         )

    #     print()
    #     print("=" * 60)
    #     print("GENIEE TRAINING")
    #     print("=" * 60)

    #     print(
    #         f"Device          : {self.device}"
    #     )

    #     print(
    #         f"Learning rate   : {self.learning_rate}"
    #     )

    #     print(
    #         f"Epochs          : {epochs}"
    #     )

    #     print("=" * 60)

    #     for epoch in range(
    #         1,
    #         epochs + 1
    #     ):

    #         self.current_epoch = epoch

    #         # ----------------------------------------------
    #         # Train
    #         # ----------------------------------------------

    #         train_loss = (
    #             self.train_epoch()
    #         )

    #         # ----------------------------------------------
    #         # Validation
    #         # ----------------------------------------------

    #         validation_loss = (
    #             self.validate()
    #         )

    #         # ----------------------------------------------
    #         # Display
    #         # ----------------------------------------------

    #         print(
    #             f"Epoch {epoch}/{epochs} "
    #             f"| train_loss={train_loss:.4f} "
    #             f"| val_loss={validation_loss:.4f}"
    #         )

    #         # ----------------------------------------------
    #         # Save checkpoint
    #         # ----------------------------------------------

    #         if (
    #             save_every > 0
    #             and epoch % save_every == 0
    #         ):

    #             self.save_checkpoint(
    #                 epoch
    #             )

    #     print()
    #     print(
    #         "Training completed."
    #     )
        # ======================================================
    # Train
    # ======================================================

    def train(
        self,
        epochs,
        save_every=1,
        early_stopping_patience=2
    ):

        if epochs <= 0:

            raise ValueError(
                "epochs must be greater than zero."
            )

        if early_stopping_patience < 0:

            raise ValueError(
                "early_stopping_patience "
                "cannot be negative."
            )

        print()
        print("=" * 60)
        print("GENIEE TRAINING")
        print("=" * 60)

        print(
            f"Device              : {self.device}"
        )

        print(
            f"Learning rate        : {self.learning_rate}"
        )

        print(
            f"Epochs               : {epochs}"
        )

        print(
            f"Early stopping      : "
            f"{early_stopping_patience}"
        )

        print("=" * 60)

        epochs_without_improvement = 0

        for epoch in range(
            1,
            epochs + 1
        ):

            self.current_epoch = epoch

            # ----------------------------------------------
            # Train
            # ----------------------------------------------

            train_loss = (
                self.train_epoch()
            )

            # ----------------------------------------------
            # Validation
            # ----------------------------------------------

            validation_loss = (
                self.validate()
            )

            # ----------------------------------------------
            # Display
            # ----------------------------------------------

            print(
                f"Epoch {epoch}/{epochs} "
                f"| train_loss={train_loss:.4f} "
                f"| val_loss={validation_loss:.4f}"
            )

            # ----------------------------------------------
            # Save regular checkpoint
            # ----------------------------------------------

            if (
                save_every > 0
                and epoch % save_every == 0
            ):

                self.save_checkpoint(
                    epoch
                )

            # ----------------------------------------------
            # Check validation improvement
            # ----------------------------------------------

            if validation_loss < self.best_validation_loss:

                self.best_validation_loss = (
                    validation_loss
                )

                self.best_epoch = epoch

                epochs_without_improvement = 0

                self.save_best_checkpoint()

                print(
                    f"✓ New best model "
                    f"| val_loss={validation_loss:.4f}"
                )

            else:

                epochs_without_improvement += 1

                print(
                    f"No validation improvement "
                    f"({epochs_without_improvement}/"
                    f"{early_stopping_patience})"
                )

            # ----------------------------------------------
            # Early stopping
            # ----------------------------------------------

            if (
                early_stopping_patience > 0
                and epochs_without_improvement
                >= early_stopping_patience
            ):

                print()
                print(
                    "Early stopping triggered."
                )

                print(
                    f"Best epoch : "
                    f"{self.best_epoch}"
                )

                print(
                    f"Best val loss : "
                    f"{self.best_validation_loss:.4f}"
                )

                break

        print()
        print(
            "Training completed."
        )

        print(
            f"Best epoch      : "
            f"{self.best_epoch}"
        )

        print(
            f"Best val loss   : "
            f"{self.best_validation_loss:.4f}"
        )

    # ======================================================
    # Save checkpoint
    # ======================================================

        # ======================================================
    # Save best checkpoint
    # ======================================================

    def save_best_checkpoint(self):

        checkpoint_path = (
            self.checkpoint_dir
            / "geniee_best.pt"
        )

        checkpoint = {

            "epoch": self.current_epoch,

            "global_step": (
                self.global_step
            ),

            "model_state_dict": (
                self.model.state_dict()
            ),

            "optimizer_state_dict": (
                self.optimizer.state_dict()
            ),

            "train_losses": (
                self.train_losses
            ),

            "validation_losses": (
                self.validation_losses
            ),

            "learning_rate": (
                self.learning_rate
            ),

            "best_validation_loss": (
                self.best_validation_loss
            )
        }

        torch.save(
            checkpoint,
            checkpoint_path
        )

        print(
            f"Best checkpoint saved: "
            f"{checkpoint_path}"
        )

        return checkpoint_path
    def save_checkpoint(
        self,
        epoch=None
    ):

        if epoch is None:

            epoch = self.current_epoch

        checkpoint_path = (
            self.checkpoint_dir
            / f"geniee_epoch_{epoch}.pt"
        )

        checkpoint = {

            "epoch": epoch,

            "global_step": (
                self.global_step
            ),

            "model_state_dict": (
                self.model.state_dict()
            ),

            "optimizer_state_dict": (
                self.optimizer.state_dict()
            ),

            "train_losses": (
                self.train_losses
            ),

            "validation_losses": (
                self.validation_losses
            ),

            "learning_rate": (
                self.learning_rate
            )
        }

        torch.save(
            checkpoint,
            checkpoint_path
        )

        print(
            f"Checkpoint saved: "
            f"{checkpoint_path}"
        )

        return checkpoint_path

    # ======================================================
    # Load checkpoint
    # ======================================================

    def load_checkpoint(
        self,
        checkpoint_path
    ):

        checkpoint_path = Path(
            checkpoint_path
        )

        if not checkpoint_path.exists():

            raise FileNotFoundError(
                f"Checkpoint not found: "
                f"{checkpoint_path}"
            )

        checkpoint = torch.load(
            checkpoint_path,
            map_location=self.device
        )

        self.model.load_state_dict(
            checkpoint[
                "model_state_dict"
            ]
        )

        self.optimizer.load_state_dict(
            checkpoint[
                "optimizer_state_dict"
            ]
        )

        self.current_epoch = (
            checkpoint.get(
                "epoch",
                0
            )
        )

        self.global_step = (
            checkpoint.get(
                "global_step",
                0
            )
        )

        self.train_losses = (
            checkpoint.get(
                "train_losses",
                []
            )
        )

        self.validation_losses = (
            checkpoint.get(
                "validation_losses",
                []
            )
        )

        print(
            f"Checkpoint loaded: "
            f"{checkpoint_path}"
        )

        return checkpoint