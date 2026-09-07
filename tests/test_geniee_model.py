import torch

from model.config import GenieeConfig
from model.geniee_model import GenieeModel


def test_geniee_model_forward_pass():

    # --------------------------------------------------
    # Use central configuration
    # --------------------------------------------------

    config = GenieeConfig()

    print()
    print("Geniee Configuration:")
    print("vocab_size     :", config.vocab_size)
    print("max_seq_len    :", config.max_seq_len)
    print("d_model        :", config.d_model)
    print("num_heads      :", config.num_heads)
    print("head_dim       :", config.head_dim)
    print("ffn_hidden_dim :", config.ffn_hidden_dim)
    print("num_layers     :", config.num_layers)
    print("dropout        :", config.dropout)

    # --------------------------------------------------
    # Create Geniee model
    # --------------------------------------------------

    model = GenieeModel(
        config=config
    )

    # --------------------------------------------------
    # Sample input
    #
    # Shape:
    # [batch_size, sequence_length]
    # --------------------------------------------------

    batch_size = 2
    sequence_length = 16

    input_ids = torch.randint(
        low=0,
        high=config.vocab_size,
        size=(
            batch_size,
            sequence_length
        )
    )

    # --------------------------------------------------
    # Forward pass
    # --------------------------------------------------

    logits = model(input_ids)

    # --------------------------------------------------
    # Print shapes
    # --------------------------------------------------

    print()
    print("Input IDs shape :", input_ids.shape)
    print("Logits shape    :", logits.shape)

    # --------------------------------------------------
    # Expected:
    #
    # [batch_size, sequence_length, vocab_size]
    # --------------------------------------------------

    expected_shape = (
        batch_size,
        sequence_length,
        config.vocab_size
    )

    assert logits.shape == expected_shape

    # --------------------------------------------------
    # Check numerical validity
    # --------------------------------------------------

    assert torch.isfinite(logits).all()

    # --------------------------------------------------
    # Check output is not all zeros
    # --------------------------------------------------

    assert not torch.all(
        logits == 0
    )


def test_geniee_model_batch_processing():

    # --------------------------------------------------
    # Central configuration
    # --------------------------------------------------

    config = GenieeConfig()

    # --------------------------------------------------
    # Create model
    # --------------------------------------------------

    model = GenieeModel(
        config=config
    )

    # --------------------------------------------------
    # Test different batch sizes
    # --------------------------------------------------

    for batch_size in [1, 2, 4]:

        sequence_length = 8

        input_ids = torch.randint(
            low=0,
            high=config.vocab_size,
            size=(
                batch_size,
                sequence_length
            )
        )

        # --------------------------------------------------
        # Forward pass
        # --------------------------------------------------

        logits = model(input_ids)

        # --------------------------------------------------
        # Verify shape
        # --------------------------------------------------

        assert logits.shape == (
            batch_size,
            sequence_length,
            config.vocab_size
        )

        # --------------------------------------------------
        # Verify valid numbers
        # --------------------------------------------------

        assert torch.isfinite(
            logits
        ).all()


def test_geniee_model_parameter_count():

    # --------------------------------------------------
    # Central configuration
    # --------------------------------------------------

    config = GenieeConfig()

    # --------------------------------------------------
    # Create model
    # --------------------------------------------------

    model = GenieeModel(
        config=config
    )

    # --------------------------------------------------
    # Count trainable parameters
    # --------------------------------------------------

    total_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    print()
    print(
        "Geniee trainable parameters:",
        f"{total_parameters:,}"
    )

    # --------------------------------------------------
    # Basic validation
    # --------------------------------------------------

    assert total_parameters > 0


def test_invalid_heads():

    # --------------------------------------------------
    # d_model = 512
    # 512 cannot be divided evenly by 7
    # --------------------------------------------------

    try:

        GenieeConfig(
            d_model=512,
            num_heads=7
        )

        assert False, "Expected ValueError"

    except ValueError:

        assert True



# import torch

# from model.config import GenieeConfig
# from model.geniee_model import GenieeModel


# def test_geniee_model_forward_pass():

#     # --------------------------------------------------
#     # Configuration
#     # --------------------------------------------------

#     config = GenieeConfig(
#         vocab_size=32_000,
#         max_seq_len=512,
#         d_model=512,
#         num_heads=8,
#         ffn_hidden_dim=2_048,
#         num_layers=8,
#         dropout=0.0
#     )

#     # --------------------------------------------------
#     # Create Geniee model
#     # --------------------------------------------------

#     model = GenieeModel(
#         config=config
#     )

#     # --------------------------------------------------
#     # Create sample token IDs
#     #
#     # Shape:
#     # [batch_size, sequence_length]
#     # --------------------------------------------------

#     batch_size = 2
#     sequence_length = 16

#     input_ids = torch.randint(
#         low=0,
#         high=config.vocab_size,
#         size=(
#             batch_size,
#             sequence_length
#         )
#     )

#     # --------------------------------------------------
#     # Forward pass
#     # --------------------------------------------------

#     logits = model(
#         input_ids
#     )

#     # --------------------------------------------------
#     # Print shapes
#     # --------------------------------------------------

#     print()
#     print("Input IDs shape :", input_ids.shape)
#     print("Logits shape    :", logits.shape)

#     # --------------------------------------------------
#     # Expected shape:
#     #
#     # [batch_size, sequence_length, vocab_size]
#     # --------------------------------------------------

#     expected_shape = (
#         batch_size,
#         sequence_length,
#         config.vocab_size
#     )

#     assert logits.shape == expected_shape

#     # --------------------------------------------------
#     # Check that logits contain valid numbers
#     # --------------------------------------------------

#     assert torch.isfinite(logits).all()

#     # --------------------------------------------------
#     # Check model output is not all zeros
#     # --------------------------------------------------

#     assert not torch.all(
#         logits == 0
#     )


# def test_geniee_model_batch_processing():

#     # --------------------------------------------------
#     # Create configuration
#     # --------------------------------------------------

#     config = GenieeConfig(
#         vocab_size=32_000,
#         max_seq_len=512,
#         d_model=512,
#         num_heads=8,
#         ffn_hidden_dim=2_048,
#         num_layers=8,
#         dropout=0.0
#     )

#     # --------------------------------------------------
#     # Create model
#     # --------------------------------------------------

#     model = GenieeModel(
#         config=config
#     )

#     # --------------------------------------------------
#     # Test different batch sizes
#     # --------------------------------------------------

#     for batch_size in [1, 2, 4]:

#         sequence_length = 8

#         input_ids = torch.randint(
#             low=0,
#             high=config.vocab_size,
#             size=(
#                 batch_size,
#                 sequence_length
#             )
#         )

#         logits = model(
#             input_ids
#         )

#         # --------------------------------------------------
#         # Verify output
#         # --------------------------------------------------

#         assert logits.shape == (
#             batch_size,
#             sequence_length,
#             config.vocab_size
#         )

#         assert torch.isfinite(
#             logits
#         ).all()


# def test_geniee_model_parameter_count():

#     # --------------------------------------------------
#     # Configuration
#     # --------------------------------------------------

#     config = GenieeConfig()

#     # --------------------------------------------------
#     # Create model
#     # --------------------------------------------------

#     model = GenieeModel(
#         config=config
#     )

#     # --------------------------------------------------
#     # Count trainable parameters
#     # --------------------------------------------------

#     total_parameters = sum(
#         parameter.numel()
#         for parameter in model.parameters()
#         if parameter.requires_grad
#     )

#     print()
#     print(
#         "Geniee trainable parameters:",
#         f"{total_parameters:,}"
#     )

#     # --------------------------------------------------
#     # Basic validation
#     # --------------------------------------------------

#     assert total_parameters > 0