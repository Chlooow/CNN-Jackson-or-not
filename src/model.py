# src/model.py
# CNN MODELS DEFINITION

import torch
import torch.nn as nn


class MyCNN(nn.Module):
    """
    Modèle CNN standard
    Input : (3, 128, 128)
    """
    def __init__(self, num_classes=2, dropout=0.5):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),   # 64x64

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),   # 32x32

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),   # 16x16
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 16 * 16, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


class MyCNNOptimized(nn.Module):
    """
    Modèle CNN simplifié (moins de paramètres)
    """
    def __init__(self, num_classes=2, dropout=0.5):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),   # 64x64

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),   # 32x32
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 32 * 32, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


def get_model(model_name="base", device=None):
    """
    Retourne le modèle choisi
    """
    if model_name == "base":
        model = MyCNN()
    elif model_name == "optimized":
        model = MyCNNOptimized()
    else:
        raise ValueError("model_name must be 'base' or 'optimized'")

    if device:
        model = model.to(device)

    return model
