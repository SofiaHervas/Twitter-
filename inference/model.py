import torch
import torch.nn as nn
from pathlib import Path

class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, output_dim, pad_idx):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, x):
        emb = self.embedding(x)
        _, (h, _) = self.lstm(emb)
        h = torch.cat([h[-2], h[-1]], dim=1)
        h = self.dropout(h)
        return self.fc(h)

def load_model(vocab_size, output_dim, pad_idx):
    model = LSTMModel(vocab_size, 100, 128, output_dim, pad_idx)
    path = Path(__file__).parent.parent / 'model' / 'emotion_model.pt'
    model.load_state_dict(torch.load(path, map_location='cpu'))
    model.eval()
    return model


