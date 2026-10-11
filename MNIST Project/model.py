import torch.nn as nn
import torch.nn.functional as F

class SimpleClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        # Input: 784 pixels, Output: 128 hidden features
        self.layer1 = nn.Linear(in_features=784, out_features=128)
        # Input: 128 hidden features, Output: 10 digit classes (0-9)
        self.layer2 = nn.Linear(in_features=128, out_features=10)

    def forward(self, x):
        # Flatten the image tensor from [batch_size, 1, 28, 28] to [batch_size, 784]
        x = x.view(-1, 784)

        x = self.layer1(x)
        x = F.relu(x)
        x = self.layer2(x)
        return x