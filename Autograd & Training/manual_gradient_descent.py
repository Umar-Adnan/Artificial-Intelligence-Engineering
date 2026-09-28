import torch

# Setup
features = torch.tensor([2.0])
target = torch.tensor([10.0])
weights = torch.tensor([0.5], requires_grad=True)
learning_rate = 0.1

print("=== Manual Weight Update ===")
for epoch in range(5):
    # 1. Forward & 2. Loss
    prediction = features * weights
    loss = (prediction - target) ** 2

    # 3. Backward
    loss.backward()

    # 4.Manual Update & 5. Manual Zero Grad
    # We MUST use torch.no_grad() here. We are modifying the weights,
    # but we don't want Autograd to track this specific mathematical step.
    with torch.no_grad():
        weights -= learning_rate * weights.grad  # Subtracting gradient * lr
        weights.grad.zero_()                     # Wiping the gradient clean

    print(f"Epoch {epoch+1} | Loss: {loss.item():.2f} | New Weight: {weights.item():.2f}")