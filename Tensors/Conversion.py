import torch
import numpy as np

# 1. Create a standard NumPy array (e.g., cleaned data from Pandas)
numpy_array = np.array([1.5, 2.5, 3.5, 4.5])
print("=== Original NumPy Array ===")
print(numpy_array, "\n")

# 2. The Memory Share Method (Standard Practice)
# Fast, efficient, but linked(Shallow Copy).
shared_tensor = torch.from_numpy(numpy_array)

print("=== Memory Sharing Demonstration ===")
# If we mutate the original NumPy array...
numpy_array[0] = 99.9
# The tensor updates automatically because they share the same memory location
print(f"NumPy Array: {numpy_array}")
print(f"Shared Tensor: {shared_tensor}\n")

# 3. The Deep Copy Method
# Slower, uses more RAM, but safely isolated.
isolated_tensor = torch.tensor(numpy_array)

# If we mutate the array again...
numpy_array[1] = 88.8
# The isolated tensor remains completely unaffected
print("=== Isolated Copy Demonstration ===")
print(f"NumPy Array: {numpy_array}")
print(f"Isolated Tensor: {isolated_tensor}")