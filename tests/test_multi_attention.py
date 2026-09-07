import torch

from model.attention import MultiHeadSelfAttention


def test_multi_head_self_attention():

    # --------------------------------------------------
    # Configuration
    # --------------------------------------------------

    batch_size = 2
    sequence_length = 5
    d_model = 512
    num_heads = 8

    # --------------------------------------------------
    # Create attention
    # --------------------------------------------------

    attention = MultiHeadSelfAttention(
        d_model=d_model,
        num_heads=num_heads
    )

    # --------------------------------------------------
    # Create random input
    # --------------------------------------------------

    X = torch.randn(
        batch_size,
        sequence_length,
        d_model
    )

    # --------------------------------------------------
    # Forward pass
    # --------------------------------------------------

    output = attention(X)

    # --------------------------------------------------
    # Print shapes
    # --------------------------------------------------

    print()
    print("Input shape :", X.shape)
    print("Output shape:", output.shape)

    # --------------------------------------------------
    # Validate
    # --------------------------------------------------

    assert output.shape == (
        batch_size,
        sequence_length,
        d_model
    )