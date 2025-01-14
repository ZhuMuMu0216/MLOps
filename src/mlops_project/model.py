import torch
from torch import nn


class ResNet18(nn.Module):
    def __init__(self, num_classes=1):
        super(ResNet18, self).__init__()
        # Use pretrained ResNet18 and modify for binary classification
        self.model = torch.hub.load("pytorch/vision:v0.10.0", "resnet18", pretrained=True)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)
