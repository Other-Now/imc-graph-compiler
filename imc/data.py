"""CIFAR-10 loader without torchvision. Run `python scripts/get_cifar.py` once
to create data/cifar10.npz (uint8 NCHW images + labels)."""
import os

import numpy as np
import torch

PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "cifar10.npz")
MEAN = np.array([0.4914, 0.4822, 0.4465], dtype=np.float32).reshape(1, 3, 1, 1)
STD = np.array([0.2470, 0.2435, 0.2616], dtype=np.float32).reshape(1, 3, 1, 1)


def _load(split):
    d = np.load(PATH)
    x = d[f"x_{split}"].astype(np.float32) / 255.0
    return torch.from_numpy((x - MEAN) / STD), torch.from_numpy(d[f"y_{split}"])


def train_set():
    return _load("train")


def test_set():
    return _load("test")


def available():
    return os.path.exists(PATH)
