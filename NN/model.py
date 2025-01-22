import torch
import torch.nn as nn

class FullyConnectedNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes, dropout_rate=0.5):
        super(FullyConnectedNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)

        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.bn2 = nn.BatchNorm1d(hidden_dim // 2)

        self.fc3 = nn.Linear(hidden_dim // 2, hidden_dim // 4)
        self.bn3 = nn.BatchNorm1d(hidden_dim // 4)

        self.elu = nn.ELU()
        self.dropout = nn.Dropout(dropout_rate)
        self.fc4 = nn.Linear(hidden_dim // 4, num_classes)

    def forward(self, x):
        # First Layer
        x = self.fc1(x)
        x = self.bn1(x)  # Batch Normalization
        x = self.elu(x)
        x = self.dropout(x)

        # Second Layer
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.elu(x)
        x = self.dropout(x)

        # Third Layer
        x = self.fc3(x)
        x = self.bn3(x)
        x = self.elu(x)
        x = self.dropout(x)

        # Output Layer
        x = self.fc4(x)
        return x
