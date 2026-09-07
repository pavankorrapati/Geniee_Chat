import torch

from model.attention import MultiHeadSelfAttention
def test_causal_mask():

    attention = MultiHeadSelfAttention(
        d_model=8,
        num_heads=2
    )

    X = torch.randn(
        1,
        4,
        8
    )

    output= attention(X)

    print(output)
    # print("Attention weights:")
    # print(weights[0, 0])

    # Future positions must receive zero attention

    # upper_triangle = weights[0, 0].triu(
    #     diagonal=1
    # )

    # assert torch.allclose(
    #     upper_triangle,
    #     torch.zeros_like(upper_triangle),
    #     atol=1e-6
    # )