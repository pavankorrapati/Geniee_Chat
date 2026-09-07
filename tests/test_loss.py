import torch

from training.loss import causal_language_modeling_loss


def test_causal_language_modeling_loss():

    # --------------------------------------------------
    # Configuration
    # --------------------------------------------------

    batch_size = 2
    sequence_length = 8
    vocab_size = 1000

    # --------------------------------------------------
    # Fake model output
    # --------------------------------------------------

    logits = torch.randn(
        batch_size,
        sequence_length,
        vocab_size
    )

    # --------------------------------------------------
    # Fake token IDs
    # --------------------------------------------------

    input_ids = torch.randint(
        low=0,
        high=vocab_size,
        size=(
            batch_size,
            sequence_length
        )
    )

    # --------------------------------------------------
    # Calculate loss
    # --------------------------------------------------

    loss = causal_language_modeling_loss(
        logits,
        input_ids
    )

    # --------------------------------------------------
    # Validate
    # --------------------------------------------------

    print()
    print("Loss:", loss.item())

    assert loss.ndim == 0

    assert torch.isfinite(
        loss
    )

    assert loss.item() > 0