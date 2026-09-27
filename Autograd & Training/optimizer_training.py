import torch
import torch.optim as optim

features = torch.tensor([2.0])
target = torch.tensor([10.0])
weights = torch.tensor([0.5], requires_grad=True)

# Define the optimizer and hand it the weights
optimizer = optim.SGD([weights], lr=0.1)

print("\n=== PyTorch Optimizer Update ===")
for epoch in range(3):
    # 1. Forward Pass & 2. Loss
    prediction = features * weights
    loss = (prediction - target) ** 2

    # 3. Zero Grad
    optimizer.zero_grad()

    # 4. Backward Pass
    loss.backward()

    # 5. Optimizer Step (Handles the no_grad() and math automatically)
    optimizer.step()

    print(f"Epoch {epoch+1} | Loss: {loss.item():.2f} | New Weight: {weights.item():.2f}")