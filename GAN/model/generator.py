import torch
import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, dataset, lstm_input_size, num_layers, hidden_dim, dropout):
        super().__init__()
        self.lstm_size = lstm_input_size
        self.embedding_dim = lstm_input_size
        self.num_layers = num_layers
        # self.hidden_dim = hidden_dim

        # test_use
        self.hidden_dim = lstm_input_size

        n_vocab = len(dataset.uniq_words)
        self.embedding = nn.Embedding(
            num_embeddings=n_vocab,
            embedding_dim=self.embedding_dim,
        )
        self.lstm = nn.LSTM(
            input_size=self.lstm_size,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            dropout=dropout,
        )
        self.fc = nn.Linear(self.lstm_size, n_vocab)

    def forward(self, x, prev_state):
        print(x.shape)
        embed = self.embedding(x)
        print(embed.shape)
        output, state = self.lstm(embed, prev_state)
        logits = self.fc(output)

        return logits, state

    def init_state(self, sequence_length):
        return (torch.zeros(self.num_layers, sequence_length, self.lstm_size),
                torch.zeros(self.num_layers, sequence_length, self.lstm_size))
