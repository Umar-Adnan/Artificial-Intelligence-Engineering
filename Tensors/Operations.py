import torch

tensor_a = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
tensor_b = torch.tensor([[10.0, 10.0], [10.0, 10.0]])

# 1. Element-Wise Arithmetic
# Multiplies position 0,0 with position 0,0; position 0,1 with 0,1, etc.
element_wise_mul = tensor_a * tensor_b
print("=== Element-Wise Multiplication ===")
print(element_wise_mul, "\n")

# 2. Reduction Operations
# No dimension specified: squashes the entire tensor into a single scalar
total_sum = tensor_a.sum()
print("=== Complete Reduction ===")
print(f"Total Sum: {total_sum.item()}\n")

# Dimension specified (dim=0 reduces rows, dim=1 reduces columns)
# Taking the mean across columns (dim=1)
column_mean = tensor_a.mean(dim=1)
print("=== Dimension Reduction (Mean of each row) ===")
print(f"Mean values: {column_mean}\n")

# 3. In-Place Operations
print("=== In-Place Mutation ===")
print(f"Before: {tensor_a}")
tensor_a.add_(100) # Adds 100 directly to tensor_a's memory location
print(f"After : {tensor_a}")