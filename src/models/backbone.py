"""
CNN Backbone for DermaMNIST Continual Learning.
Supports class-incremental head expansion.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DermaCNN(nn.Module):
    """
    Lightweight CNN backbone for 28x28 RGB skin lesion images.
    The classifier head grows dynamically as new tasks are added.
    """

    def __init__(self, num_classes: int = 0):
        super().__init__()

        # Feature extractor
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),         # 14x14
            nn.Dropout2d(0.25),

            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),         # 7x7
            nn.Dropout2d(0.25),

            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((3, 3)),  # 3x3
        )

        self.feature_dim = 128 * 3 * 3  # = 1152

        # MLP neck
        self.neck = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.feature_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
        )

        # Classifier head (expanded per task)
        self.classifier = nn.Linear(256, num_classes) if num_classes > 0 else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.features(x)
        feat = self.neck(feat)
        if self.classifier is not None:
            return self.classifier(feat)
        return feat

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Return embedding before classifier."""
        feat = self.features(x)
        return self.neck(feat)

    def expand_classifier(self, new_classes: int) -> None:
        """
        Add new_classes output neurons to the classifier head.
        Used for class-incremental learning — old weights are preserved.
        """
        if self.classifier is None:
            self.classifier = nn.Linear(256, new_classes)
        else:
            old_weight = self.classifier.weight.data
            old_bias = self.classifier.bias.data
            old_out = old_weight.shape[0]
            total = old_out + new_classes

            new_layer = nn.Linear(256, total)
            new_layer.weight.data[:old_out] = old_weight
            new_layer.bias.data[:old_out] = old_bias
            # Xavier init for new neurons
            nn.init.xavier_uniform_(new_layer.weight.data[old_out:])
            nn.init.zeros_(new_layer.bias.data[old_out:])
            self.classifier = new_layer

    @property
    def num_classes(self) -> int:
        return self.classifier.out_features if self.classifier is not None else 0


def build_model(num_classes: int = 0) -> DermaCNN:
    return DermaCNN(num_classes=num_classes)
