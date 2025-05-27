import argparse
import torch
from .model import load_model
from .utils import encode_text, load_vocab, load_label_encoder

def main():
    parser = argparse.ArgumentParser(description="Emotion Classifier CLI")
    parser.add_argument('--input', type=str, help='Text input for emotion prediction')
    parser.add_argument('--kaggle', action='store_true', help='Display Kaggle ID')
    args = parser.parse_args()

    if args.kaggle:
        print("sofahervs")
        return

    if args.input:
        print("Input received:", args.input) 
        vocab = load_vocab()
        label_encoder = load_label_encoder()
        model = load_model(vocab_size=len(vocab), output_dim=len(label_encoder.classes_), pad_idx=vocab['<PAD>'])

        encoded = encode_text([args.input], vocab)
        with torch.no_grad():
            output = model(encoded)
            pred = torch.argmax(output, dim=1).item()
            emotion = label_encoder.inverse_transform([pred])[0]
            print(emotion)
