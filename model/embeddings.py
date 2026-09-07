import torch
import torch.nn as nn


class TokenAndPositionEmbedding(nn.Module):
    """
    Combines token embeddings and positional embeddings.

    Input:
        Token IDs
        [batch_size, sequence_length]

    Output:
        Embedded tokens
        [batch_size, sequence_length, d_model]
    """

    def __init__(
        self,
        vocab_size,
        max_seq_len,
        d_model,
        dropout=0.0
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.d_model = d_model

        # -----------------------------------------
        # Token embedding
        # -----------------------------------------

        self.token_embedding = nn.Embedding(
            vocab_size,
            d_model
        )

        # -----------------------------------------
        # Position embedding
        # -----------------------------------------

        self.position_embedding = nn.Embedding(
            max_seq_len,
            d_model
        )

        # -----------------------------------------
        # Dropout
        # -----------------------------------------

        self.dropout = nn.Dropout(
            dropout
        )

    def forward(self, input_ids):

        # -----------------------------------------
        # Input shape:
        #
        # [batch_size, sequence_length]
        # -----------------------------------------

        batch_size, sequence_length = input_ids.shape

        # -----------------------------------------
        # Validate sequence length
        # -----------------------------------------

        if sequence_length > self.max_seq_len:
            raise ValueError(
                f"Sequence length {sequence_length} "
                f"exceeds max_seq_len "
                f"{self.max_seq_len}"
            )

        # -----------------------------------------
        # Token embeddings
        # -----------------------------------------

        token_embeddings = self.token_embedding(
            input_ids
        )

        # Shape:
        #
        # [batch_size, sequence_length, d_model]

        # -----------------------------------------
        # Position IDs
        # -----------------------------------------

        position_ids = torch.arange(
            sequence_length,
            device=input_ids.device
        )

        # -----------------------------------------
        # Position embeddings
        # -----------------------------------------

        position_embeddings = self.position_embedding(
            position_ids
        )

        # Shape:
        #
        # [sequence_length, d_model]

        # -----------------------------------------
        # Combine token + position
        # -----------------------------------------

        embeddings = (
            token_embeddings
            + position_embeddings
        )

        # -----------------------------------------
        # Dropout
        # -----------------------------------------

        embeddings = self.dropout(
            embeddings
        )

        return embeddings