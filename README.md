# My AI Engineering Journey & PyTorch MNIST Project

## Overview
As part of my ongoing AI and Machine Learning studies, this repository reflects my progression from core Python fundamentals (including OOP, abstraction, and encapsulation) to building practical, modular Deep Learning models using PyTorch. 

## What I Learned in AI Engineering
- **Python for AI/ML:** Mastered object-oriented programming principles to structure scalable applications.
- **Data Science Foundations:** Utilized libraries like Pandas for data manipulation and Matplotlib for clear, structured visualizations.
- **PyTorch Fundamentals:** 
  - Managing tensors and hardware acceleration (routing computations between CPU and GPU/CUDA).
  - Designing Neural Networks by subclassing `torch.nn.Module`.
  - Engineering data pipelines using `torchvision.datasets` and the `DataLoader` engine.
  - Implementing the standard 5-step optimization loop (Forward pass, Loss calculation, Zeroing gradients, Backpropagation, and Weight updates).
- **Version Control Best Practices:** Structuring modular repositories and utilizing Git/GitHub for clean project management directly from the terminal.

## The Project: Modular MNIST Handwritten Digit Classifier
This project is a clean, multi-file implementation of a neural network that classifies handwritten digits (0-9) from the standard MNIST dataset. It serves as a foundational "Hello World" to deep learning architecture.

### Architecture & File Structure
The project is decoupled into separate components following standard industry practices:

- **`model.py`**: Defines the `SimpleClassifier` blueprint. It utilizes a 2-layer Feed-Forward Neural Network (Multi-Layer Perceptron) using `nn.Linear` and `ReLU` activation, flattening 28x28 pixel images into a 784-feature 1D array.
- **`dataset.py`**: Handles raw data downloading, tensor conversion (`transforms.ToTensor()`), and batching via `DataLoader` for both training and testing datasets.
- **`train.py`**: The main execution engine. It uses `CrossEntropyLoss` and the highly efficient `Adam` optimizer to train the network over 3 epochs. It automatically detects and routes math to the GPU if available, eventually saving the optimized weights to `simple_classifier.pth`.
- **`evaluate.py`**: A testing script that locks the network (`torch.no_grad()`), loads the saved `.pth` weights, calculates the final accuracy against 10,000 unseen test images, and uses Matplotlib to generate a visual grid of predictions versus actual handwritten shapes.

### Results
- **Final Accuracy:** **96.92%**
- Despite flattening the images and losing 2D spatial awareness, the network successfully learned to identify pixel patterns using pure linear algebra and non-linear activations. 

### Next Steps
With this modular baseline established, the architecture is ready to be upgraded with `nn.Conv2d` (Convolutional Neural Networks) to break the 99% accuracy barrier, or to be wrapped in a FastAPI endpoint for web deployment.