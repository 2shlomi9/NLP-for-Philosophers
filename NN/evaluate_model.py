import torch
from sklearn.metrics import classification_report, accuracy_score

def evaluate_model(model, x_test, y_test, class_names, device='cpu'):
    """
    Evaluate a Fully Connected Neural Network (FNN).
    """
    # Convert to PyTorch tensors
    x_test = torch.tensor(x_test, dtype=torch.float32).to(device)
    y_test = torch.tensor(y_test, dtype=torch.long).to(device)

    # Make predictions
    model.eval()
    with torch.no_grad():
        y_pred = model(x_test)
        y_pred_classes = torch.argmax(y_pred, dim=1)

    # Accuracy
    accuracy = accuracy_score(y_test.cpu(), y_pred_classes.cpu())

    # Classification report
    print("Classification Report:")
    print(classification_report(y_test.cpu(), y_pred_classes.cpu(), target_names=class_names))
    print(f"Test Accuracy: {accuracy * 100:.2f}%")

    return accuracy
