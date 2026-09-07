from model.config import GenieeConfig


def test_default_config():

    config = GenieeConfig()

    assert config.vocab_size == 32_000

    assert config.max_seq_len == 512

    assert config.d_model == 512

    assert config.num_heads == 8

    assert config.ffn_hidden_dim == 2_048

    assert config.num_layers == 8

    assert config.dropout == 0.1

    assert config.head_dim == 64