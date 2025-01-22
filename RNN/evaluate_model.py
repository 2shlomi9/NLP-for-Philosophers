import torch
from sklearn.metrics import classification_report, accuracy_score
import pandas as pd

def evaluate_model(model, val_loader, label_encoder, device):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for texts, labels in val_loader:
            texts, labels = texts.to(device), labels.to(device)
            outputs = model(texts)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Generate classification report
    report = classification_report(
        all_labels, all_preds, target_names=label_encoder.classes_, output_dict=True
    )
    df_report = pd.DataFrame(report).transpose()
    df_report = df_report.round(2)

    # Display the classification report as a table
    print("\nClassification Report:")
    print(df_report)

    # Test accuracy
    accuracy = accuracy_score(all_labels, all_preds)
    print(f"\nTest Accuracy: {accuracy:.2%}")