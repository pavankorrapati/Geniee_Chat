import torch
import torch.nn.functional as F


def causal_language_modeling_loss(
    logits,
    input_ids,
    ignore_index=-100
):
    """
    Calculates next-token prediction loss.

    Parameters
    ----------
    logits:
        Model predictions.

        Shape:
            [batch_size, sequence_length, vocab_size]

    input_ids:
        Original token IDs.

        Shape:
            [batch_size, sequence_length]

    Returns
    -------
    torch.Tensor
        Scalar cross-entropy loss.
    """

    # --------------------------------------------------
    # Shift logits
    #
    # We don't need to predict the first token.
    #
    # Example:
    #
    # input:
    #   I am learning Python
    #
    # predictions:
    #   am learning Python <EOS>
    # --------------------------------------------------

    shift_logits = logits[:, :-1, :]

    # --------------------------------------------------
    # Shift targets
    # --------------------------------------------------

    shift_labels = input_ids[:, 1:]

    # --------------------------------------------------
    # Flatten
    #
    # CrossEntropyLoss expects:
    #
    # predictions:
    # [N, vocab_size]
    #
    # labels:
    # [N]
    # --------------------------------------------------

    shift_logits = shift_logits.reshape(
        -1,
        shift_logits.size(-1)
    )

    shift_labels = shift_labels.reshape(
        -1
    )

    # --------------------------------------------------
    # Calculate Cross Entropy
    # --------------------------------------------------

    loss = F.cross_entropy(
        shift_logits,
        shift_labels,
        ignore_index=ignore_index
    )

    return loss