
import torch
from evaluate_model import evaluate_model

def train(model, train_loader, val_loader, optimizer, scheduler, device, epochs=4):
    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []

    for epoch in range(epochs):
        model.train()
        total_train_loss = 0
        correct_train = 0
        total_train = 0
        print(f"Epoch {epoch+1}/{epochs}:")

        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            logits = outputs.logits

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            total_train_loss += loss.item()
            predictions = torch.argmax(logits, dim=-1)
            correct_train += (predictions == batch['labels']).sum().item()
            total_train += batch['labels'].size(0)

        train_loss = total_train_loss / len(train_loader)
        train_acc = correct_train / total_train
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        print(f"Epoch {epoch+1}/{epochs}: Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.4f}")

        eval_loss, eval_acc = evaluate_model(model, val_loader, device)
        val_losses.append(eval_loss)
        val_accuracies.append(eval_acc)
        print(f"Validation Loss: {eval_loss:.4f}, Validation Accuracy: {eval_acc:.4f}")
        
    return train_losses, train_accuracies, val_losses, val_accuracies