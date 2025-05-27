import argparse
import torch
from .model import load_model
from .utils import encode_text, load_vocab

def main():
    parser = argparse.ArgumentParser(description="Emotion Classifier CLI")
    parser.add_argument('--input', type=str, help='Text input for emotion prediction')
    parser.add_argument('--kaggle', action='store_true', help='Display Kaggle ID')
    args = parser.parse_args()

    if args.kaggle:
        print("SofiaHervas")  # Cambia esto por tu username real de Kaggle si es otro
        return

    if args.input:
        vocab = load_vocab()

        # Modelo entrenado con 6 emociones
        model = load_model(
            vocab_size=len(vocab),
            output_dim=6,  # ← Fuerza 6 clases (importante)
            pad_idx=vocab['<PAD>']
        )

        encoded = encode_text([args.input], vocab)
        with torch.no_grad():
            output = model(encoded)
            pred = torch.argmax(output, dim=1).item()

        # Mapeo manual a etiquetas
        class_labels = ['anger', 'fear', 'joy', 'love', 'sadness', 'surprise']
        emotion = class_labels[pred]
        print(emotion)
