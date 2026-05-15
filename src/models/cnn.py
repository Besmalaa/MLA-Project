"""
CNN backbone for DermaMNIST continual learning.
Simple but effective architecture for 28x28 RGB images.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class DermaCNN(nn.Module):
    """
    Small CNN backbone for DermaMNIST (28x28 RGB, 7 classes).
    Used as the shared feature extractor across all CL methods.
    """

    def __init__(self, num_classes: int = 7, hidden_dim: int = 256):
        super().__init__()
        self.num_classes = num_classes

        # Feature extractor
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),   # 14x14
            nn.Dropout2d(0.1),

            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),   # 7x7
            nn.Dropout2d(0.1),

            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((3, 3)),  # 3x3
        )

        self.feature_dim = 128 * 3 * 3  # 1152

        # Classifier head
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.feature_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.features(x)
        return self.classifier(feats)

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Return flattened feature vector (before classifier)."""
        feats = self.features(x)
        return feats.view(feats.size(0), -1)


def build_model(num_classes: int = 7, device: str = "cpu") -> DermaCNN:
    model = DermaCNN(num_classes=num_classes)
    return model.to(device)
