import torch
import torch.nn as nn

class LSTMWithAttention(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, bidirectional, dropout):
        super(LSTMWithAttention, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.LSTM(embedding_dim, hidden_dim, num_layers=n_layers, bidirectional=bidirectional, dropout=dropout, batch_first=True)
        self.fc = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.attention = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, 1)

    def forward(self, text, text_lengths):
        embedded = self.dropout(self.embedding(text))
        
        # Move text_lengths to CPU
        text_lengths = text_lengths.cpu()
        
        # Pack the sequences
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, text_lengths, batch_first=True, enforce_sorted=False)
        
        packed_output, (hidden, cell) = self.rnn(packed_embedded)
        output, output_lengths = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)

        # Apply attention mechanism
        attention_weights = torch.softmax(self.attention(output), dim=1)
        attended_output = torch.sum(output * attention_weights, dim=1)

        return self.fc(attended_output)
