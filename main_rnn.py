import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from RNN.model import RNNModelWithAttention
from RNN.evaluate_model import evaluate_model
from RNN.test_processor import TextProcessor, TextDataset, load_data
from RNN.train_model import train_model


# Load and preprocess data
file_path = 'Dataset/philosophy_data_edit.csv'
model_path = "Models/rnn_model.pth"  

X_train, X_val, y_train, y_val, label_encoder = load_data(file_path)
processor = TextProcessor()
X_train_processed = processor.fit_transform(X_train)
X_val_processed = processor.transform(X_val)

train_dataset = TextDataset(X_train_processed, y_train)
val_dataset = TextDataset(X_val_processed, y_val)
train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=512, shuffle=False)

# Define model and training parameters
vocab_size = 10000
embed_dim = 128
hidden_dim = 256
output_dim = len(label_encoder.classes_)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = RNNModelWithAttention(vocab_size, embed_dim, hidden_dim, output_dim)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Train the model
model = train_model(model, train_loader, val_loader, criterion, optimizer, epochs=10, device=device)

# Save the trained model
torch.save(model.state_dict(), model_path)
print(f"Model saved to {model_path}")

# Evaluate the model
evaluate_model(model, val_loader, label_encoder, device)

