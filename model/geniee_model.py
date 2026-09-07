import torch
import torch.nn as nn

from model.config import GenieeConfig
from model.embeddings import TokenAndPositionEmbedding
from model.transformer_block import TransformerBlock


class GenieeModel(nn.Module):
    """
    Geniee decoder-only Transformer language model.

    Input:

        input_ids
        [batch_size, sequence_length]

    Optional:

        attention_mask
        [batch_size, sequence_length]

        1 = real token
        0 = padding

    Output:

        logits
        [batch_size, sequence_length, vocab_size]
    """

    def __init__(
        self,
        config=None,
    ):
        super().__init__()

        # --------------------------------------------------
        # Configuration
        # --------------------------------------------------

        if config is None:

            config = GenieeConfig()

        self.config = config

        # --------------------------------------------------
        # Token + position embeddings
        # --------------------------------------------------

        self.embeddings = (
            TokenAndPositionEmbedding(
                vocab_size=config.vocab_size,
                max_seq_len=config.max_seq_len,
                d_model=config.d_model,
                dropout=config.dropout,
            )
        )

        # --------------------------------------------------
        # Transformer blocks
        # --------------------------------------------------

        self.transformer_blocks = (
            nn.ModuleList(
                [
                    TransformerBlock(
                        d_model=config.d_model,
                        num_heads=config.num_heads,
                        ffn_hidden_dim=(
                            config.ffn_hidden_dim
                        ),
                        dropout=config.dropout,
                    )
                    for _ in range(
                        config.num_layers
                    )
                ]
            )
        )

        # --------------------------------------------------
        # Final LayerNorm
        # --------------------------------------------------

        self.final_norm = nn.LayerNorm(
            config.d_model
        )

        # --------------------------------------------------
        # Language model head
        # --------------------------------------------------

        self.lm_head = nn.Linear(
            config.d_model,
            config.vocab_size,
            bias=False,
        )

    def forward(
        self,
        input_ids,
        attention_mask=None,
    ):
        """
        Parameters
        ----------
        input_ids:
            Tensor:

                [B, S]

        attention_mask:
            Optional tensor:

                [B, S]

            1 = real token
            0 = padding

        Returns
        -------
        logits:
            [B, S, V]
        """

        # --------------------------------------------------
        # Validate input dimensions
        # --------------------------------------------------

        if input_ids.dim() != 2:

            raise ValueError(
                "input_ids must have shape "
                "[batch_size, sequence_length]."
            )

        batch_size = input_ids.size(0)

        sequence_length = input_ids.size(1)

        # --------------------------------------------------
        # Validate sequence length
        # --------------------------------------------------

        if sequence_length > (
            self.config.max_seq_len
        ):

            raise ValueError(
                "Input sequence length "
                f"({sequence_length}) exceeds "
                "model max_seq_len "
                f"({self.config.max_seq_len})."
            )

        # --------------------------------------------------
        # Validate attention mask
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
            # Convert to same device as input.
            # ------------------------------------------------

            attention_mask = (
                attention_mask.to(
                    device=input_ids.device
                )
            )

        # --------------------------------------------------
        # 1. Token + position embeddings
        # --------------------------------------------------

        x = self.embeddings(
            input_ids
        )

        # --------------------------------------------------
        # 2. Transformer blocks
        # --------------------------------------------------

        for block in (
            self.transformer_blocks
        ):

            x = block(
                x,
                attention_mask=attention_mask,
            )

        # --------------------------------------------------
        # 3. Final LayerNorm
        # --------------------------------------------------

        x = self.final_norm(
            x
        )

        # --------------------------------------------------
        # 4. Language model head
        # --------------------------------------------------

        logits = self.lm_head(
            x
        )

        # --------------------------------------------------
        # Output:
        #
        # [B, S, vocab_size]
        # --------------------------------------------------

        return logits