import pickle
import torch
from pathlib import Path
import os

def load_vocab():
    path = Path(__file__).parent.parent / 'model' / 'vocab.pkl'
    with open(path, 'rb') as f:
        return pickle.load(f)

def load_label_encoder():
    path = Path(__file__).parent.parent / 'model' / 'label_encoder.pkl'
    with open(path, 'rb') as f:
        return pickle.load(f)

def encode_text(texts, vocab, max_length=100):
    sequences = []
    for text in texts:
        tokens = text.lower().split()
        seq = [vocab.get(tok, vocab['<OOV>']) for tok in tokens]
        if len(seq) < max_length:
            seq += [vocab['<PAD>']] * (max_length - len(seq))
        else:
            seq = seq[:max_length]
        sequences.append(seq)
    return torch.tensor(sequences)
