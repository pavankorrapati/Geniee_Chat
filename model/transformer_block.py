import torch
import torch.nn as nn

from model.attention import MultiHeadSelfAttention
from model.feed_forward import FeedForward


class TransformerBlock(nn.Module):
    """
    Single pre-norm Transformer decoder block.

    Architecture:

        Input
          │
          ├─────────────── Residual
          │
          ▼
       LayerNorm
          │
          ▼
    Multi-Head Causal
       Attention
          │
          ▼
       Add Residual
          │
          ├─────────────── Residual
          │
          ▼
       LayerNorm
          │
          ▼
     Feed Forward
          │
          ▼
       Add Residual
          │
          ▼
        Output

    The attention layer receives the padding mask so that
    PAD tokens cannot be used as attention keys.
    """

    def __init__(
        self,
        d_model,
        num_heads,
        ffn_hidden_dim,
        dropout=0.0,
    ):
        super().__init__()

        # --------------------------------------------------
        # LayerNorm before attention
        # --------------------------------------------------

        self.norm1 = nn.LayerNorm(
            d_model
        )

        # --------------------------------------------------
        # Multi-head self-attention
        # --------------------------------------------------

        self.attention = (
            MultiHeadSelfAttention(
                d_model=d_model,
                num_heads=num_heads,
                dropout=dropout,
            )
        )

        # --------------------------------------------------
        # LayerNorm before feed-forward
        # --------------------------------------------------

        self.norm2 = nn.LayerNorm(
            d_model
        )

        # --------------------------------------------------
        # Feed-forward network
        # --------------------------------------------------

        self.feed_forward = (
            FeedForward(
                d_model=d_model,
                hidden_dim=ffn_hidden_dim,
                dropout=dropout,
            )
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
            [batch_size, sequence_length, d_model]

        attention_mask:
            [batch_size, sequence_length]

            1 = real token
            0 = padding

        Returns
        -------
        X:
            [batch_size, sequence_length, d_model]
        """

        # --------------------------------------------------
        # 1. Pre-normalization
        # --------------------------------------------------

        normalized_X = self.norm1(
            X
        )

        # --------------------------------------------------
        # 2. Attention
        #
        # Pass attention_mask through.
        # --------------------------------------------------

        attention_output = (
            self.attention(
                normalized_X,
                attention_mask=attention_mask,
            )
        )

        # --------------------------------------------------
        # 3. Attention residual
        # --------------------------------------------------

        X = X + attention_output

        # --------------------------------------------------
        # 4. Pre-normalization before FFN
        # --------------------------------------------------

        normalized_X = self.norm2(
            X
        )

        # --------------------------------------------------
        # 5. Feed-forward
        # --------------------------------------------------

        feed_forward_output = (
            self.feed_forward(
                normalized_X
            )
        )

        # --------------------------------------------------
        # 6. FFN residual
        # --------------------------------------------------

        X = X + feed_forward_output

        return X