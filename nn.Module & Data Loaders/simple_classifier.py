import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleClassifier(nn.Module):
    # 1. The Blueprint (__init__)
    # This is where you declare all the layers your network will use.
    # PyTorch automatically tracks the weights inside these layers.
    def __init__(self):
        # Mandatory: initializes the underlying nn.Module mechanics
        super().__init__()

        # Layer 1: takes 784 input features (e.g., a flattened 28x28 MNIST image), outputs 128
        self.layer1 = nn.Linear(in_features=784, out_features=128)

        # Layer 2: takes the 128 features from layer 1, outputs 10 (e.g., digits 0-9)
        self.layer2 = nn.Linear(in_features=128, out_features=10)

    # 2. The Execution (forward)
    # This defines exactly how the input data (x) flows through the layers built above.
    def forward(self, x):
        # Pass input through layer 1
        x = self.layer1(x)

        # Apply an activation function (ReLU) to introduce non-linearity
        x = F.relu(x)

        # Pass through the final output layer
        x = self.layer2(x)

        return x

# Instantiate the network
model = SimpleClassifier()

# Printing the model shows you the exact architecture and layer connections
print(model)
# Save the model's learned weights
torch.save(model.state_dict(), "simple_classifier.pth")