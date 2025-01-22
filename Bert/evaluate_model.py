from sklearn.metrics import accuracy_score, classification_report
import torch

def evaluate_model(model, loader, device):
    model.eval()
    total_eval_loss = 0
    correct_eval = 0
    total_eval = 0

    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            logits = outputs.logits

            total_eval_loss += loss.item()
            predictions = torch.argmax(logits, dim=-1)
            correct_eval += (predictions == batch['labels']).sum().item()
            total_eval += batch['labels'].size(0)

    eval_loss = total_eval_loss / len(loader)
    eval_acc = correct_eval / total_eval
    return eval_loss, eval_acc

def final_evaluation(model, test_loader, device):
    print("\nFinal Model Evaluation on Test Data:")
    model.eval()
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for batch in test_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)

            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(batch['labels'].cpu().numpy())

    test_accuracy = accuracy_score(all_labels, all_predictions)
    print(f"Test Accuracy: {test_accuracy:.4f}")

    report = classification_report(all_labels, all_predictions, target_names=['Class 0', 'Class 1', 'Class 2', 'Class 3'])
    print("\nClassification Report:")
    print(report)
