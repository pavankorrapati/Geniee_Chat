import torch

from model.embeddings import TokenAndPositionEmbedding


def test_token_and_position_embedding():

    # -----------------------------------------
    # Configuration
    # -----------------------------------------

    batch_size = 2
    sequence_length = 16
    vocab_size = 32000
    max_seq_len = 512
    d_model = 512

    # -----------------------------------------
    # Create embedding module
    # -----------------------------------------

    embedding = TokenAndPositionEmbedding(
        vocab_size=vocab_size,
        max_seq_len=max_seq_len,
        d_model=d_model
    )

    # -----------------------------------------
    # Create token IDs
    # -----------------------------------------

    input_ids = torch.randint(
        low=0,
        high=vocab_size,
        size=(
            batch_size,
            sequence_length
        )
    )

    # -----------------------------------------
    # Forward pass
    # -----------------------------------------

    output = embedding(
        input_ids
    )

    # -----------------------------------------
    # Print shapes
    # -----------------------------------------

    print()
    print("Input IDs shape :", input_ids.shape)
    print("Output shape    :", output.shape)

    # -----------------------------------------
    # Validate
    # -----------------------------------------

    assert input_ids.shape == (
        batch_size,
        sequence_length
    )

    assert output.shape == (
        batch_size,
        sequence_length,
        d_model
    )

def test_embedding_rejects_long_sequence():

    embedding = TokenAndPositionEmbedding(
        vocab_size=10000,
        max_seq_len=16,
        d_model=512
    )

    input_ids = torch.randint(
        0,
        10000,
        (2, 17)
    )

    try:

        embedding(input_ids)

        assert False, (
            "Expected ValueError"
        )

    except ValueError:

        assert True    