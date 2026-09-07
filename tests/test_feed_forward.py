import torch

from model.feed_forward import FeedForward


def test_feed_forward():

    # -----------------------------------------
    # Configuration
    # -----------------------------------------

    batch_size = 2
    sequence_length = 5
    d_model = 512
    hidden_dim = 2048

    # -----------------------------------------
    # Create FFN
    # -----------------------------------------

    feed_forward = FeedForward(
        d_model=d_model,
        hidden_dim=hidden_dim
    )

    # -----------------------------------------
    # Create input
    # -----------------------------------------

    X = torch.randn(
        batch_size,
        sequence_length,
        d_model
    )

    # -----------------------------------------
    # Forward pass
    # -----------------------------------------

    output = feed_forward(X)

    # -----------------------------------------
    # Print shapes
    # -----------------------------------------

    print()
    print("Input shape :", X.shape)
    print("Output shape:", output.shape)

    # -----------------------------------------
    # Validate
    # -----------------------------------------

    assert output.shape == (
        batch_size,
        sequence_length,
        d_model
    )