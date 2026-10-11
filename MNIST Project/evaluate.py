import torch
import matplotlib.pyplot as plt
from model import SimpleClassifier
from dataset import get_dataloaders

# 1. Load the Data and the Model
_, test_loader = get_dataloaders(batch_size=32)

model = SimpleClassifier()
model.load_state_dict(torch.load("simple_classifier.pth", weights_only=True))

# Lock the network for testing (disables training-specific layer behaviors)
model.eval()

# 2. Test the Overall Accuracy
correct_predictions = 0
total_samples = 0

# Disable the computational graph to save memory and speed up execution
with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)

        # Get the index of the highest probability prediction (the predicted digit)
        _, predicted = torch.max(outputs, 1)

        total_samples += labels.size(0)
        correct_predictions += (predicted == labels).sum().item()

accuracy = (correct_predictions / total_samples) * 100
print(f"Final Model Accuracy on 10,000 test images: {accuracy:.2f}%\n")

# 3. Visualize a Single Batch
with torch.no_grad():
    # Grab just one batch of 32 images and labels
    dataiter = iter(test_loader)
    images, labels = next(dataiter)

    outputs = model(images)
    _, predicted = torch.max(outputs, 1)

# Set up a Matplotlib grid to display the first 6 images from the batch
fig, axes = plt.subplots(2, 3, figsize=(10, 6))

for i, ax in enumerate(axes.flat):
    # Convert the PyTorch tensor back to a standard NumPy array for plotting
    img = images[i].squeeze().numpy()

    ax.imshow(img, cmap='gray')
    ax.set_title(f"Pred: {predicted[i].item()} | True: {labels[i].item()}")
    ax.axis('off')

plt.tight_layout()
plt.show()