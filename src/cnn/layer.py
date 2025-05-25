import torch.nn as nn
import torch.nn.functional as F
from cnn.bitnet import BinaryConv2d, BinaryLinear


class BinaryCNN(nn.Module):
    """
    Binary Convolutional Neural Network (BinaryCNN)

    This model is designed for image classification (e.g. MNIST) using binarized convolutional and linear layers.
    The weights are binarized using a sign function and scaled (Hybrid Binary), allowing the model to maintain
    good performance while being memory- and energy-efficient.

    Architecture Summary:
        - 6 convolutional layers using BinaryConv2d + BatchNorm + ReLU
        - MaxPooling is applied after conv1, conv2, conv4, and conv6
        - Adaptive average pooling to reduce feature map to 1x1
        - Fully connected binary linear layer followed by classifier

    Output:
        - Final output is 10-class logits (for MNIST: digits 0–9)
    """

    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = BinaryConv2d(1, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)

        self.conv2 = BinaryConv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        self.conv3 = BinaryConv2d(64, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        self.conv4 = BinaryConv2d(128, 128, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(128)

        self.conv5 = BinaryConv2d(128, 256, 3, padding=1)
        self.bn5 = nn.BatchNorm2d(256)

        self.conv6 = BinaryConv2d(256, 256, 3, padding=1)
        self.bn6 = nn.BatchNorm2d(256)

        self.pool = nn.MaxPool2d(2, 2)
        self.globalpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.5)

        self.fc1 = nn.Sequential(
            BinaryLinear(256, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.5)
        )
        self.fc2 = BinaryLinear(256, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))  # Conv1 → MaxPool
        x = self.pool(F.relu(self.bn2(self.conv2(x))))  # Conv2 → MaxPool
        x = F.relu(self.bn3(self.conv3(x)))  # Conv3 (no pooling)
        x = self.pool(F.relu(self.bn4(self.conv4(x))))  # Conv4 → MaxPool
        x = F.relu(self.bn5(self.conv5(x)))  # Conv5
        x = self.pool(F.relu(self.bn6(self.conv6(x))))  # Conv6 → MaxPool
        x = self.globalpool(x)  # Global average pool
        x = x.view(x.size(0), -1)  # Flatten
        x = self.dropout(x)
        x = self.fc1(x)
        return self.fc2(x)
