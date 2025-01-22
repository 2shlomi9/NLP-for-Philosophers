import torch
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report

def evaluate_model(model, x_test, y_test, classes_names):
    if not isinstance(x_test, torch.Tensor):
        x_test = torch.tensor(x_test, dtype=torch.float32)
    if not isinstance(y_test, torch.Tensor):
        y_test = torch.tensor(np.argmax(y_test, axis=1), dtype=torch.long)  # One-hot to class indices
    
    model.eval()
    with torch.no_grad():
        y_pred = model(x_test)
        y_pred_classes = torch.argmax(y_pred, dim=1)

    accuracy = (y_pred_classes == y_test).sum().item() / len(y_test)

    # Calculate precision, recall, and F1-score for each class
    precision = precision_score(y_test, y_pred_classes, average=None)
    recall = recall_score(y_test, y_pred_classes, average=None)
    f1 = f1_score(y_test, y_pred_classes, average=None)

    print(f"Test Accuracy: {accuracy * 100:.2f}%")

    # Print metrics for each class
    print("Classification Report:")
    print(classification_report(y_test, y_pred_classes, target_names=[f"{classes_names[i]}" for i in range(len(precision))]))

    return precision, recall, f1
