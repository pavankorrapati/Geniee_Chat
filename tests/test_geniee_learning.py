import torch

from model.config import GenieeConfig
from model.geniee_model import GenieeModel
from training.loss import causal_language_modeling_loss


def test_geniee_can_learn_tiny_dataset():

    # --------------------------------------------------
    # Tiny configuration
    #
    # IMPORTANT:
    # We use a tiny model here for testing.
    # We don't use the full 8-layer Geniee model.
    # --------------------------------------------------

    config = GenieeConfig(
        vocab_size=32,
        max_seq_len=16,
        d_model=64,
        num_heads=4,
        ffn_hidden_dim=128,
        num_layers=2,
        dropout=0.0
    )

    # --------------------------------------------------
    # Create model
    # --------------------------------------------------

    model = GenieeModel(
        config=config
    )

    model.train()

    # --------------------------------------------------
    # Tiny repeated training example
    # --------------------------------------------------

    input_ids = torch.tensor([
        [1, 2, 3, 4, 5, 6],
        [1, 2, 3, 4, 5, 6],
        [1, 2, 3, 4, 5, 6],
        [1, 2, 3, 4, 5, 6]
    ])

    # --------------------------------------------------
    # Optimizer
    # --------------------------------------------------

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=0.001
    )

    # --------------------------------------------------
    # Initial loss
    # --------------------------------------------------

    logits = model(input_ids)

    initial_loss = causal_language_modeling_loss(
        logits,
        input_ids
    )

    initial_loss_value = initial_loss.item()

    # --------------------------------------------------
    # Train
    # --------------------------------------------------

    for step in range(100):

        optimizer.zero_grad()

        logits = model(
            input_ids
        )

        loss = causal_language_modeling_loss(
            logits,
            input_ids
        )

        loss.backward()

        optimizer.step()

    # --------------------------------------------------
    # Final loss
    # --------------------------------------------------

    logits = model(
        input_ids
    )

    final_loss = causal_language_modeling_loss(
        logits,
        input_ids
    )

    final_loss_value = final_loss.item()

    print()
    print(
        "Initial loss:",
        initial_loss_value
    )

    print(
        "Final loss:",
        final_loss_value
    )

    # --------------------------------------------------
    # Validate learning
    # --------------------------------------------------

    assert final_loss_value < initial_loss_value