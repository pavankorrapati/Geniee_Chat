from dataclasses import dataclass


@dataclass
class GenieeConfig:
    """Central configuration for the Geniee decoder-only Transformer."""

    vocab_size: int = 32_000
    max_seq_len: int = 512
    d_model: int = 512
    num_heads: int = 8
    ffn_hidden_dim: int = 2048
    num_layers: int = 8
    dropout: float = 0.1

    def __post_init__(self):
        if self.vocab_size <= 0:
            raise ValueError("vocab_size must be greater than 0")
        if self.max_seq_len <= 0:
            raise ValueError("max_seq_len must be greater than 0")
        if self.d_model <= 0:
            raise ValueError("d_model must be greater than 0")
        if self.num_heads <= 0:
            raise ValueError("num_heads must be greater than 0")
        if self.d_model % self.num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        if self.ffn_hidden_dim <= 0:
            raise ValueError("ffn_hidden_dim must be greater than 0")
        if self.num_layers <= 0:
            raise ValueError("num_layers must be greater than 0")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be between 0 and 1")

    @property
    def head_dim(self):
        return self.d_model // self.num_heads
