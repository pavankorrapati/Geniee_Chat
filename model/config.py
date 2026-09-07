from dataclasses import dataclass


@dataclass
class GenieeConfig:
    """
    Central configuration for the Geniee language model.
    """

    # --------------------------------------------------
    # Vocabulary
    # --------------------------------------------------

    vocab_size: int = 32_000

    # --------------------------------------------------
    # Context
    # --------------------------------------------------

    max_seq_len: int = 512

    # --------------------------------------------------
    # Transformer dimensions
    # --------------------------------------------------

    d_model: int = 512

    num_heads: int = 8

    ffn_hidden_dim: int = 2_048

    num_layers: int = 8

    # --------------------------------------------------
    # Regularization
    # --------------------------------------------------

    dropout: float = 0.1

    # --------------------------------------------------
    # Validation
    # --------------------------------------------------

    def __post_init__(self):

        if self.d_model <= 0:
            raise ValueError(
                "d_model must be greater than 0"
            )

        if self.num_heads <= 0:
            raise ValueError(
                "num_heads must be greater than 0"
            )

        if self.d_model % self.num_heads != 0:
            raise ValueError(
                "d_model must be divisible by num_heads"
            )

        if self.ffn_hidden_dim <= 0:
            raise ValueError(
                "ffn_hidden_dim must be greater than 0"
            )

        if self.num_layers <= 0:
            raise ValueError(
                "num_layers must be greater than 0"
            )

        if self.vocab_size <= 0:
            raise ValueError(
                "vocab_size must be greater than 0"
            )

        if self.max_seq_len <= 0:
            raise ValueError(
                "max_seq_len must be greater than 0"
            )

        if not 0.0 <= self.dropout < 1.0:
            raise ValueError(
                "dropout must be between 0 and 1"
            )

    @property
    def head_dim(self):
        """
        Dimension of each attention head.
        """

        return self.d_model // self.num_heads
#     def test_invalid_heads():
#         try:

#             GenieeConfig(
#                 d_model=512,
#                 num_heads=7
#             )

#             assert False, (
#                 "Expected ValueError"
#             )

#         except ValueError:

#             assert True



# # from dataclasses import dataclass


# # @dataclass
# # class GenieeConfig:

# #     # Vocabulary
# #     vocab_size: int = 32000

# #     # Context length
# #     max_sequence_length: int = 512

# #     # Transformer
# #     num_layers: int = 8
# #     num_heads: int = 8

# #     # Hidden representation
# #     embedding_dim: int = 512

# #     # Feed-forward network
# #     feed_forward_dim: int = 2048

# #     # Dropout
# #     dropout: float = 0.1

# #     # Special tokens
# #     pad_token_id: int = 0
# #     bos_token_id: int = 2
# #     eos_token_id: int = 3