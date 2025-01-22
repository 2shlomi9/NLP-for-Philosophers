import torch
import torch.optim as optim
import torch.nn as nn
from sklearn.model_selection import train_test_split
import numpy as np
# Helper function for calculating accuracy
def calculate_accuracy(y_pred, y_true):
    """Calculate accuracy by comparing predicted and true labels."""
    y_pred_classes = torch.argmax(y_pred, dim=1)
    accuracy = (y_pred_classes == y_true).sum().item() / len(y_true)
    return accuracy

def train_fnn(x_data, y_data, model, num_classes, epochs=100, lr=0.001, batch_size=32, device='cpu'):
    """
    Train a Fully Connected Neural Network (FNN).
    """
    # Split into train and validation sets
    x_train, x_val, y_train, y_val = train_test_split(x_data, y_data, test_size=0.2, random_state=42)

    # Convert to PyTorch tensors
    x_train = torch.tensor(x_train, dtype=torch.float32).to(device)
    y_train = torch.tensor(y_train, dtype=torch.long).to(device)
    x_val = torch.tensor(x_val, dtype=torch.float32).to(device)
    y_val = torch.tensor(y_val, dtype=torch.long).to(device)

    # Count occurrences of each class
    class_counts = np.bincount(y_train)
    total_samples = len(y_train)

    # Compute class weights
    class_weights = total_samples / (num_classes * class_counts)
    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Optional: Implement a learning rate scheduler
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)

    # Training loop
    for epoch in range(epochs):
        model.train()

        total_train_loss = 0.0
        total_train_accuracy = 0.0
        num_batches = 0

        for i in range(0, len(x_train), batch_size):
            x_batch = x_train[i:i + batch_size]
            y_batch = y_train[i:i + batch_size]
            optimizer.zero_grad()
            y_pred = model(x_batch)
            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()
            total_train_accuracy += calculate_accuracy(y_pred, y_batch)
            num_batches += 1

        # Calculate average train loss and accuracy
        avg_train_loss = total_train_loss / num_batches
        avg_train_accuracy = total_train_accuracy / num_batches

        # Validation phase
        model.eval()
        with torch.no_grad():
            y_val_pred = model(x_val)
            val_loss = criterion(y_val_pred, y_val).item()
            val_accuracy = calculate_accuracy(y_val_pred, y_val)

        # Step the learning rate scheduler
        scheduler.step()

        # Print training progress
        if epoch % 10 == 0 or epoch == epochs - 1:
            print(
                f"Epoch {epoch}/{epochs}, Train Loss: {avg_train_loss:.4f}, Train Accuracy: {avg_train_accuracy * 100:.2f}%, "
                f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy * 100:.2f}%"
            )

    print("Training complete!")
    return model
