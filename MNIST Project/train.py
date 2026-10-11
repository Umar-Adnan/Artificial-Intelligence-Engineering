import torch
import torch.nn as nn
import torch.optim as optim
from model import SimpleClassifier
from dataset import get_dataloaders

# 1. Hardware Configuration
# Automatically routes math to the GPU if available, drastically speeding up training
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training on device: {device}")

# 2. Initialize components and send model to the selected hardware
train_loader, test_loader = get_dataloaders(batch_size=32)
model = SimpleClassifier().to(device)

# 3. Define the Loss Function and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 4. The Training Loop
epochs = 3 

print("Starting training...")
for epoch in range(epochs):
    running_loss = 0.0
    
    for images, labels in train_loader:
        # Move the data to the same hardware as the model
        images, labels = images.to(device), labels.to(device)
        
        # Step 1: Forward Pass
        predictions = model(images)
        
        # Step 2: Calculate the Loss
        loss = criterion(predictions, labels)
        
        # Step 3: Zero the Gradients
        optimizer.zero_grad()
        
        # Step 4: Backward Pass
        loss.backward()
        
        # Step 5: Update Weights
        optimizer.step()
        
        running_loss += loss.item()
        
    average_loss = running_loss / len(train_loader)
    print(f"Epoch {epoch+1}/{epochs} Complete - Average Loss: {average_loss:.4f}")

# 5. Save the Final Optimized Weights
torch.save(model.state_dict(), "simple_classifier.pth")
print("Training complete. Model weights saved to simple_classifier.pth")