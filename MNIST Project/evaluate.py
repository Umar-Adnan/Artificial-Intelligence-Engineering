import torch
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from model import SimpleClassifier
from dataset import get_dataloaders

# 1. Load the Data and the Model
_, test_loader = get_dataloaders(batch_size=32)

model = SimpleClassifier()
model.load_state_dict(torch.load("simple_classifier.pth", weights_only=True))
model.eval()

# 2. Test the Overall Accuracy
correct_predictions = 0
total_samples = 0

with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        
        total_samples += labels.size(0)
        correct_predictions += (predicted == labels).sum().item()

accuracy = (correct_predictions / total_samples) * 100
print(f"Final Model Accuracy on 10,000 test images: {accuracy:.2f}%\n")

# 3. Visualize a Single Batch (Black and Red Theme)
with torch.no_grad():
    dataiter = iter(test_loader)
    images, labels = next(dataiter)
    outputs = model(images)
    _, predicted = torch.max(outputs, 1)

# Enable dark mode for the entire plot
plt.style.use('dark_background')

# Create a custom colormap where 0 (empty space) is black and 1 (ink) is vibrant red
black_red_cmap = LinearSegmentedColormap.from_list('black_red', ['#000000', '#ff3333'])

fig, axes = plt.subplots(2, 3, figsize=(10, 6))
fig.patch.set_facecolor('#0a0a0a') # Set the window background to deep black

for i, ax in enumerate(axes.flat):
    img = images[i].squeeze().numpy()
    
    # Display the image using the custom black/red colormap
    ax.imshow(img, cmap=black_red_cmap)
    
    # Color the text red (or white if you want contrast, but red fits the theme)
    ax.set_title(f"Pred: {predicted[i].item()} | True: {labels[i].item()}", 
                 color='#ff3333', 
                 fontweight='bold', 
                 fontsize=12)
    
    # Hide the axes borders
    ax.axis('off')

plt.tight_layout()
plt.show()