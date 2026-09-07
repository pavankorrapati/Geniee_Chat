import torch

from model.attention import SelfAttention


def test_self_attention():

    # -----------------------------------------
    # Configuration
    # -----------------------------------------

    batch_size = 2
    sequence_length = 5
    d_model = 8

    # -----------------------------------------
    # Create attention
    # -----------------------------------------

    attention = SelfAttention(
        d_model=d_model
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

    output = attention(X)
    # print(f"OUTPUT: {output}")
    # print(f"WEIGHT: {weights}")

    # -----------------------------------------
    # Print shapes
    # -----------------------------------------

    print()
    print("X shape      :", X.shape)
    print("Output shape :", output.shape)

    # -----------------------------------------
    # Validate
    # -----------------------------------------

    assert X.shape == (
        batch_size,
        sequence_length,
        d_model
    )

    assert output.shape == (
        batch_size,
        sequence_length,
        d_model
    )



# import torch

# from model.attention import SelfAttention


# def test_self_attention():

#     batch_size = 2
#     sequence_length = 5
#     d_model = 8

#     attention = SelfAttention(
#         d_model=d_model
#     )
    

#     X = torch.randn(
#         batch_size,
#         sequence_length,
#         d_model
#     )

#     output = attention(X)

#     print()
#     print("Input shape :", X.shape)
#     print("Output shape:", output.shape)

#     assert output.shape == (
#         batch_size,
#         sequence_length,
#         d_model
#     )