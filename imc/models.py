"""The three models the compiler is tested on. All are plain nn.Modules that
torch.fx.symbolic_trace can capture without help."""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    """CIFAR-10 MLP. The 3072x256 first layer needs 12 crossbars at 256x256."""

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(3 * 32 * 32, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class SmallCNN(nn.Module):
    """Four 3x3 conv+BN+ReLU layers, two max-pools, two FC layers."""

    def __init__(self):
        super().__init__()
        def block(cin, cout):
            return nn.Conv2d(cin, cout, 3, padding=1), nn.BatchNorm2d(cout)
        self.conv1, self.bn1 = block(3, 32)
        self.conv2, self.bn2 = block(32, 32)
        self.conv3, self.bn3 = block(32, 64)
        self.conv4, self.bn4 = block(64, 64)
        self.pool = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool(F.relu(self.bn4(self.conv4(x))))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class TransformerBlock(nn.Module):
    """One single-head, post-norm encoder block over a [tokens, dim] input.
    Used only for the fp32 import path (it has runtime-by-runtime matmuls,
    softmax and layernorm, which do not map onto weight-stationary crossbars)."""

    def __init__(self, dim=64, hidden=128):
        super().__init__()
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.ln1 = nn.LayerNorm(dim)
        self.ff1 = nn.Linear(dim, hidden)
        self.ff2 = nn.Linear(hidden, dim)
        self.ln2 = nn.LayerNorm(dim)
        self.scale = 1.0 / math.sqrt(dim)

    def forward(self, x):
        att = (self.q(x) @ self.k(x).t()) * self.scale
        att = F.softmax(att, dim=-1)
        x = self.ln1(x + self.o(att @ self.v(x)))
        return self.ln2(x + self.ff2(F.relu(self.ff1(x))))


def build(name):
    return {"mlp": MLP, "cnn": SmallCNN, "transformer": TransformerBlock}[name]()
