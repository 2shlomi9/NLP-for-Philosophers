import torch
import torch.nn as nn

class AttentionLayer(nn.Module):
    def __init__(self, hidden_dim):
        super(AttentionLayer, self).__init__()
        self.attention_weights = nn.Linear(hidden_dim * 2, 1, bias=False)

    def forward(self, lstm_output, hidden_state):
        # lstm_output: [batch_size, seq_len, hidden_dim * 2]
        # hidden_state: [batch_size, hidden_dim * 2]
        hidden_state = hidden_state.unsqueeze(2)  # [batch_size, hidden_dim * 2, 1]
        attention_scores = self.attention_weights(lstm_output)  # [batch_size, seq_len, 1]
        attention_weights = torch.softmax(attention_scores, dim=1)  # [batch_size, seq_len, 1]
        context_vector = torch.sum(attention_weights * lstm_output, dim=1)  # [batch_size, hidden_dim * 2]
        return context_vector, attention_weights

class RNNModelWithAttention(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, output_dim, dropout=0.5):
        super(RNNModelWithAttention, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=2, batch_first=True, bidirectional=True, dropout=dropout)
        self.attention = AttentionLayer(hidden_dim)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.embedding(x)  # [batch_size, seq_len, embed_dim]
        lstm_output, (hidden, _) = self.lstm(x)  # [batch_size, seq_len, hidden_dim * 2]
        hidden_state = torch.cat((hidden[-2], hidden[-1]), dim=1)  # Combine forward and backward hidden states
        context_vector, attention_weights = self.attention(lstm_output, hidden_state)  # [batch_size, hidden_dim * 2]
        context_vector = self.dropout(context_vector)
        output = self.fc(context_vector)  # [batch_size, output_dim]
        return output
