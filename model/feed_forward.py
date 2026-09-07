import torch
import torch.nn as nn


class FeedForward(nn.Module):
    """
    Feed-Forward Network used inside the Geniee Transformer block.

    Input:
        [batch_size, sequence_length, d_model]

    Output:
        [batch_size, sequence_length, d_model]
    """

    def __init__(
        self,
        d_model,
        hidden_dim,
        dropout=0.0
    ):
        super().__init__()

        # -----------------------------------------
        # First projection
        # d_model → hidden_dim
        # -----------------------------------------

        self.fc1 = nn.Linear(
            d_model,
            hidden_dim
        )

        # -----------------------------------------
        # Activation
        # -----------------------------------------

        self.activation = nn.GELU()

        # -----------------------------------------
        # Second projection
        # hidden_dim → d_model
        # -----------------------------------------

        self.fc2 = nn.Linear(
            hidden_dim,
            d_model
        )

        # -----------------------------------------
        # Dropout
        # -----------------------------------------

        self.dropout = nn.Dropout(
            dropout
        )

    def forward(self, X):

        # -----------------------------------------
        # 1. Expand representation
        # -----------------------------------------

        # print("Before FFN :", X.shape)
        X = self.fc1(X)
        # print("After fc1  :", X.shape)

        # -----------------------------------------
        # 2. Apply non-linearity
        # -----------------------------------------

        X = self.activation(X)
        # print("After GELU :", X.shape)

        # -----------------------------------------
        # 3. Project back to d_model
        # -----------------------------------------

        X = self.fc2(X)
        # print("After fc2  :", X.shape)

        # -----------------------------------------
        # 4. Dropout
        # -----------------------------------------

        X = self.dropout(X)

        return X