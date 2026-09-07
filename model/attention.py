import math

import torch
import torch.nn as nn


class SelfAttention(nn.Module):
    """
    Single-head causal self-attention.

    This class is kept mainly for educational/reference purposes.
    Geniee uses MultiHeadSelfAttention in the actual Transformer.
    """

    def __init__(
        self,
        d_model,
    ):
        super().__init__()

        self.d_model = d_model

        self.Wq = nn.Linear(
            d_model,
            d_model,
        )

        self.Wk = nn.Linear(
            d_model,
            d_model,
        )

        self.Wv = nn.Linear(
            d_model,
            d_model,
        )

        self.Wo = nn.Linear(
            d_model,
            d_model,
        )

    def forward(
        self,
        X,
        attention_mask=None,
    ):
        """
        X:
            [batch_size, sequence_length, d_model]

        attention_mask:
            [batch_size, sequence_length]

            1 = real token
            0 = padding
        """

        # --------------------------------------------------
        # Input dimensions
        # --------------------------------------------------

        batch_size = X.size(0)

        sequence_length = X.size(1)

        # --------------------------------------------------
        # 1. Q, K, V
        # --------------------------------------------------

        Q = self.Wq(X)

        K = self.Wk(X)

        V = self.Wv(X)

        # --------------------------------------------------
        # 2. Attention scores
        # --------------------------------------------------

        scores = (
            Q
            @ K.transpose(-2, -1)
        )

        # --------------------------------------------------
        # 3. Scale
        # --------------------------------------------------

        scores = (
            scores
            / math.sqrt(self.d_model)
        )

        # --------------------------------------------------
        # 4. Causal mask
        #
        # Shape:
        #
        # [sequence, sequence]
        #
        # Example:
        #
        # 1 0 0 0
        # 1 1 0 0
        # 1 1 1 0
        # 1 1 1 1
        # --------------------------------------------------

        causal_mask = torch.tril(
            torch.ones(
                sequence_length,
                sequence_length,
                device=X.device,
                dtype=torch.bool,
            )
        )

        scores = scores.masked_fill(
            ~causal_mask,
            torch.finfo(
                scores.dtype
            ).min,
        )

        # --------------------------------------------------
        # 5. Padding/key mask
        # --------------------------------------------------

        if attention_mask is not None:

            if attention_mask.dim() != 2:

                raise ValueError(
                    "attention_mask must have shape "
                    "[batch_size, sequence_length]."
                )

            if (
                attention_mask.size(0)
                != batch_size
            ):

                raise ValueError(
                    "attention_mask batch size "
                    "does not match input."
                )

            if (
                attention_mask.size(1)
                != sequence_length
            ):

                raise ValueError(
                    "attention_mask sequence length "
                    "does not match input."
                )

            key_padding_mask = (
                attention_mask
                == 0
            )

            key_padding_mask = (
                key_padding_mask
                .unsqueeze(1)
            )

            scores = scores.masked_fill(
                key_padding_mask,
                torch.finfo(
                    scores.dtype
                ).min,
            )

        # --------------------------------------------------
        # 6. Softmax
        # --------------------------------------------------

        weights = torch.softmax(
            scores,
            dim=-1,
        )

        # --------------------------------------------------
        # 7. Weighted values
        # --------------------------------------------------

        output = (
            weights
            @ V
        )

        # --------------------------------------------------
        # 8. Output projection
        # --------------------------------------------------

        output = self.Wo(
            output
        )

        return output


class MultiHeadSelfAttention(nn.Module):
    """
    Multi-head causal self-attention used by Geniee.

    Input:
        X
        [batch_size, sequence_length, d_model]

    attention_mask:
        [batch_size, sequence_length]

        1 = real token
        0 = padding

    Attention scores:
        [batch_size, num_heads, sequence_length, sequence_length]

    Two masks are applied:

        1. Causal mask
           Prevents attending to future tokens.

        2. Padding mask
           Prevents attending to PAD tokens.
    """

    def __init__(
        self,
        d_model,
        num_heads,
        dropout=0.0,
    ):
        super().__init__()

        # --------------------------------------------------
        # Validate configuration
        # --------------------------------------------------

        if d_model <= 0:

            raise ValueError(
                "d_model must be greater than zero."
            )

        if num_heads <= 0:

            raise ValueError(
                "num_heads must be greater than zero."
            )

        if d_model % num_heads != 0:

            raise ValueError(
                "d_model must be divisible "
                "by num_heads."
            )

        if not 0.0 <= dropout < 1.0:

            raise ValueError(
                "dropout must be between 0 and 1."
            )

        # --------------------------------------------------
        # Configuration
        # --------------------------------------------------

        self.d_model = d_model

        self.num_heads = num_heads

        self.head_dim = (
            d_model // num_heads
        )

        # --------------------------------------------------
        # Query projection
        # --------------------------------------------------

        self.Wq = nn.Linear(
            d_model,
            d_model,
        )

        # --------------------------------------------------
        # Key projection
        # --------------------------------------------------

        self.Wk = nn.Linear(
            d_model,
            d_model,
        )

        # --------------------------------------------------
        # Value projection
        # --------------------------------------------------

        self.Wv = nn.Linear(
            d_model,
            d_model,
        )

        # --------------------------------------------------
        # Output projection
        # --------------------------------------------------

        self.Wo = nn.Linear(
            d_model,
            d_model,
        )

        # --------------------------------------------------
        # Attention dropout
        # --------------------------------------------------

        self.dropout = nn.Dropout(
            dropout
        )

    def forward(
        self,
        X,
        attention_mask=None,
    ):
        """
        Parameters
        ----------
        X:
            Tensor with shape:

                [B, S, D]

        attention_mask:
            Optional tensor with shape:

                [B, S]

            Values:

                1 = real token
                0 = padding

        Returns
        -------
        output:
            [B, S, D]
        """

        # --------------------------------------------------
        # Input dimensions
        # --------------------------------------------------

        batch_size = X.size(0)

        sequence_length = X.size(1)

        # --------------------------------------------------
        # Validate sequence length
        # --------------------------------------------------

        if sequence_length <= 0:

            raise ValueError(
                "Sequence length must be greater than zero."
            )

        # --------------------------------------------------
        # 1. Q, K, V projections
        # --------------------------------------------------

        Q = self.Wq(X)

        K = self.Wk(X)

        V = self.Wv(X)

        # --------------------------------------------------
        # 2. Split into heads
        #
        # Before:
        #
        # [B, S, D]
        #
        # After:
        #
        # [B, S, H, head_dim]
        # --------------------------------------------------

        Q = Q.view(
            batch_size,
            sequence_length,
            self.num_heads,
            self.head_dim,
        )

        K = K.view(
            batch_size,
            sequence_length,
            self.num_heads,
            self.head_dim,
        )

        V = V.view(
            batch_size,
            sequence_length,
            self.num_heads,
            self.head_dim,
        )

        # --------------------------------------------------
        # 3. Move heads before sequence
        #
        # [B, S, H, Hd]
        #
        # ->
        #
        # [B, H, S, Hd]
        # --------------------------------------------------

        Q = Q.transpose(
            1,
            2,
        )

        K = K.transpose(
            1,
            2,
        )

        V = V.transpose(
            1,
            2,
        )

        # --------------------------------------------------
        # 4. Attention scores
        #
        # [B, H, S, Hd]
        #
        # @
        #
        # [B, H, Hd, S]
        #
        # =
        #
        # [B, H, S, S]
        # --------------------------------------------------

        scores = (
            Q
            @ K.transpose(-2, -1)
        )

        # --------------------------------------------------
        # 5. Scale
        # --------------------------------------------------

        scores = (
            scores
            / math.sqrt(self.head_dim)
        )

        # --------------------------------------------------
        # 6. Causal mask
        #
        # Prevent every position from seeing
        # future positions.
        # --------------------------------------------------

        causal_mask = torch.tril(
            torch.ones(
                sequence_length,
                sequence_length,
                device=X.device,
                dtype=torch.bool,
            )
        )

        # --------------------------------------------------
        # Broadcast:
        #
        # causal_mask:
        #
        # [S, S]
        #
        # scores:
        #
        # [B, H, S, S]
        # --------------------------------------------------

        scores = scores.masked_fill(
            ~causal_mask,
            torch.finfo(
                scores.dtype
            ).min,
        )

        # --------------------------------------------------
        # 7. Padding/key mask
        # --------------------------------------------------

        if attention_mask is not None:

            if attention_mask.dim() != 2:

                raise ValueError(
                    "attention_mask must have shape "
                    "[batch_size, sequence_length]."
                )

            if (
                attention_mask.size(0)
                != batch_size
            ):

                raise ValueError(
                    "attention_mask batch size "
                    "does not match input_ids."
                )

            if (
                attention_mask.size(1)
                != sequence_length
            ):

                raise ValueError(
                    "attention_mask sequence length "
                    "does not match input_ids."
                )

            # ------------------------------------------------
            # Convert:
            #
            # [B, S]
            #
            # into:
            #
            # [B, 1, 1, S]
            #
            # The final dimension represents KEY positions.
            # ------------------------------------------------

            key_padding_mask = (
                attention_mask
                == 0
            )

            key_padding_mask = (
                key_padding_mask
                .unsqueeze(1)
                .unsqueeze(2)
            )

            # ------------------------------------------------
            # Apply padding mask.
            #
            # This prevents every query position from
            # attending to PAD keys.
            # ------------------------------------------------

            scores = scores.masked_fill(
                key_padding_mask,
                torch.finfo(
                    scores.dtype
                ).min,
            )

        # --------------------------------------------------
        # 8. Softmax
        # --------------------------------------------------

        attention_weights = (
            torch.softmax(
                scores,
                dim=-1,
            )
        )

        # --------------------------------------------------
        # 9. Attention dropout
        # --------------------------------------------------

        attention_weights = (
            self.dropout(
                attention_weights
            )
        )

        # --------------------------------------------------
        # 10. Weighted values
        #
        # [B, H, S, S]
        #
        # @
        #
        # [B, H, S, Hd]
        #
        # =
        #
        # [B, H, S, Hd]
        # --------------------------------------------------

        context = (
            attention_weights
            @ V
        )

        # --------------------------------------------------
        # 11. Move sequence before heads
        #
        # [B, H, S, Hd]
        #
        # ->
        #
        # [B, S, H, Hd]
        # --------------------------------------------------

        context = context.transpose(
            1,
            2,
        )

        # --------------------------------------------------
        # 12. Combine heads
        #
        # [B, S, H, Hd]
        #
        # ->
        #
        # [B, S, D]
        # --------------------------------------------------

        context = (
            context
            .contiguous()
            .view(
                batch_size,
                sequence_length,
                self.d_model,
            )
        )

        # --------------------------------------------------
        # 13. Output projection
        # --------------------------------------------------

        output = self.Wo(
            context
        )

        return output