import torch

from model.transformer_block import TransformerBlock


def test_transformer_block():

    # --------------------------------------------------
    # Configuration
    # --------------------------------------------------

    batch_size = 2
    sequence_length = 16
    d_model = 512
    num_heads = 8
    ffn_hidden_dim = 2048

    # --------------------------------------------------
    # Create Transformer Block
    # --------------------------------------------------

    block = TransformerBlock(
        d_model=d_model,
        num_heads=num_heads,
        ffn_hidden_dim=ffn_hidden_dim
    )

    # --------------------------------------------------
    # Create input
    # --------------------------------------------------

    X = torch.randn(
        batch_size,
        sequence_length,
        d_model
    )

    # --------------------------------------------------
    # Forward pass
    # --------------------------------------------------

    output = block(X)

    # --------------------------------------------------
    # Print shapes
    # --------------------------------------------------

    print()
    print("Input shape :", X.shape)
    print("Output shape:", output.shape)

    # --------------------------------------------------
    # Validate output shape
    # --------------------------------------------------

    assert output.shape == (
        batch_size,
        sequence_length,
        d_model
    )