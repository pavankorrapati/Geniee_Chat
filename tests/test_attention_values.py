import torch

from model.attention import SelfAttention


# ----------------------------------
# Input
# ----------------------------------

X = torch.tensor([
    [
        [1., 2., 3., 4.],
        [5., 6., 7., 8.]
    ]
])


# ----------------------------------
# Create Attention
# d_model = 4 because each token
# contains 4 values
# ----------------------------------

attention = SelfAttention(d_model=4)


# ----------------------------------
# Print X
# ----------------------------------

print("X:")
print(X)

print("\nX shape:")
print(X.shape)


# ----------------------------------
# Print Wq
# ----------------------------------

print("\nWq:")
print(attention.Wq)

print("\nWq.weight:")
print(attention.Wq.weight)

print("\nWq.weight shape:")
print(attention.Wq.weight.shape)


# ----------------------------------
# Print bias
# ----------------------------------

print("\nWq.bias:")
print(attention.Wq.bias)


# ----------------------------------
# Calculate Q using PyTorch Linear
# ----------------------------------

print("\nQ from Linear:")

Q = attention.Wq(X)

print(Q)

print("\nQ shape:")
print(Q.shape)


# ----------------------------------
# Calculate Q manually
# ----------------------------------

print("\nQ manually:")

Q_manual = (
    X @ attention.Wq.weight.T
    + attention.Wq.bias
)

print(Q_manual)


# ----------------------------------
# Compare both
# ----------------------------------

print("\nAre they equal?")

print(torch.allclose(Q, Q_manual))

# # ----------------------------------
# # Create Wq
# # ----------------------------------

# Wq = nn.Linear(4, 4)


# # ----------------------------------
# # Print parameters
# # ----------------------------------

# print("Wq:")
# print(Wq.weight)

# print("\nBias:")
# print(Wq.bias)


# # ----------------------------------
# # Calculate Q
# # ----------------------------------

# Q = Wq(X)


# # ----------------------------------
# # Print values
# # ----------------------------------

# print("\nX:")
# print(X)

# print("\nQ:")
# print(Q)

# print("\nShapes:")
# print("X:", X.shape)
# print("Wq:", Wq.weight.shape)
# print("Q:", Q.shape)