import torch

# 1. torch.zeros(): Creates a tensor filled entirely with 0s.
# AI Use Case: Initializing network biases or creating empty padding for data.
zeros_tensor = torch.zeros((3, 3))
print("=== Tensor of Zeros ===")
print(zeros_tensor)
print(f"Shape: {zeros_tensor.shape}\n")

# 2. torch.ones(): Creates a tensor filled entirely with 1s.
# AI Use Case: Creating baseline multipliers or attention masks.
ones_tensor = torch.ones((2, 4))
print("=== Tensor of Ones ===")
print(ones_tensor)
print(f"Shape: {ones_tensor.shape}\n")

# 3. torch.rand(): Creates a tensor with random numbers from a uniform distribution [0, 1).
# AI Use Case: Initializing neural network weights before training begins.
rand_tensor = torch.rand((3, 2))
print("=== Random Tensor ===")
print(rand_tensor)
print(f"Shape: {rand_tensor.shape}")