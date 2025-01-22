from Bert.text_processor import TextDataset, load_data
from Bert.evaluate_model import evaluate, final_evaluation
from Bert.train_model import train, plot_metrics
from Bert.model import initialize_model
from torch.utils.data import DataLoader
from transformers import BertTokenizer
from plot_graph import plot_graph

def main():
    # Load and preprocess data
    file_path = 'Dataset/philosophy_data_edit.csv'
    X_train, X_val, X_test, y_train, y_val, y_test = load_data(file_path)

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    train_dataset = TextDataset(X_train, y_train, tokenizer, max_length=128)
    val_dataset = TextDataset(X_val, y_val, tokenizer, max_length=128)
    test_dataset = TextDataset(X_test, y_test, tokenizer, max_length=128)

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=3)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=3)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=3)

    # Initialize model, tokenizer, optimizer, scheduler, and device
    model, tokenizer, optimizer, scheduler, device = initialize_model(train_loader)

    # Train the model
    train_losses, train_accuracies, val_losses, val_accuracies = train(
        model, train_loader, val_loader, optimizer, scheduler, device)

    # Evaluate the model on the test set
    final_evaluation(model, test_loader, device)

    # Plot training and validation metrics
    plot_graph(train_losses, val_losses, train_accuracies, val_accuracies)