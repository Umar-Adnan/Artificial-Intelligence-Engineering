import torch

# Create a 3x4 matrix (3 rows, 4 columns)
# Imagine 3 users, each with 4 specific feature values
data_matrix = torch.tensor([
    [10.0, 20.0, 30.0, 40.0],  # User 0
    [50.0, 60.0, 70.0, 80.0],  # User 1
    [90.0, 91.0, 92.0, 93.0]   # User 2
])

print("=== Original 3x4 Tensor ===")
print(data_matrix, "\n")

# 1. Indexing a specific row (Select User 1)
user_1_data = data_matrix[1]
print("=== Row Indexing ===")
print(f"Row 1 (User 1): {user_1_data}\n")

# 2. Indexing a specific column (Select Feature 2 for all users)
# The colon ':' means "select all rows", then we specify column index 2
feature_2_all_users = data_matrix[:, 2]
print("=== Column Indexing ===")
print(f"Column 2 (Feature 2): {feature_2_all_users}\n")

# 3. Slicing a range (Select Users 0 and 1, and only their first 2 features)
subset = data_matrix[0:2, 0:2]
print("=== Range Slicing ===")
print(f"Subset:\n{subset}\n")

# 4. Extracting a specific scalar value (User 2, Feature 3)
single_value_tensor = data_matrix[2, 3]
print("=== Single Element Extraction ===")
print(f"Tensor output: {single_value_tensor}")

# CRITICAL PYTORCH METHOD: .item()
# When you have a tensor containing only a single number, use .item()
# to convert it back to a standard Python float or int.
pure_python_number = single_value_tensor.item()
print(f"Pure Python number (.item()): {pure_python_number}")