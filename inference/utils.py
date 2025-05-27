import pickle
import torch
import importlib.resources

def load_vocab():
    with importlib.resources.open_binary('inference.data', 'vocab.pkl') as f:
        return pickle.load(f)

def load_label_encoder():
    with importlib.resources.open_binary('inference.data', 'label_encoder.pkl') as f:
        return pickle.load(f)
