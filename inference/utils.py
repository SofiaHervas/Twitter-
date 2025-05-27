import pickle
import torch
import importlib.resources

def load_vocab():
    with importlib.resources.open_binary('inference.data', 'vocab.pkl') as f:
        return pickle.load(f)

def load_label_encoder():
    with importlib.resources.open_binary('inference.data', 'label_encoder.pkl') as f:
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
