"""Train the MLP and the CNN on CIFAR-10 and save state_dicts to checkpoints/.

    python -m imc.train            # both models, GPU if available
"""
import argparse
import os
import time

import torch
import torch.nn.functional as F

from . import data, models

CKPT = os.path.join(os.path.dirname(os.path.dirname(__file__)), "checkpoints")


def accuracy(model, x, y, dev, bs=1000):
    model.eval()
    hit = 0
    with torch.no_grad():
        for i in range(0, len(x), bs):
            hit += (model(x[i:i + bs].to(dev)).argmax(1).cpu() == y[i:i + bs]).sum().item()
    return hit / len(x)


def train(name, epochs, dev):
    torch.manual_seed(0)
    xtr, ytr = data.train_set()
    xte, yte = data.test_set()
    model = models.build(name).to(dev)
    opt = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9, weight_decay=5e-4)
    steps = epochs * (len(xtr) // 128)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=0.05, total_steps=steps)
    for ep in range(epochs):
        model.train()
        t0 = time.time()
        perm = torch.randperm(len(xtr))
        for i in range(0, len(xtr) - 127, 128):
            idx = perm[i:i + 128]
            xb = xtr[idx]
            flip = torch.rand(len(xb)) < 0.5            # random horizontal flip
            xb[flip] = xb[flip].flip(3)
            loss = F.cross_entropy(model(xb.to(dev)), ytr[idx].to(dev))
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
        print(f"{name} epoch {ep + 1}/{epochs} loss {loss.item():.3f} "
              f"test acc {accuracy(model, xte, yte, dev):.4f} ({time.time() - t0:.0f}s)", flush=True)
    os.makedirs(CKPT, exist_ok=True)
    torch.save(model.cpu().state_dict(), os.path.join(CKPT, f"{name}.pt"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="mlp,cnn")
    ap.add_argument("--epochs", type=int, default=15)
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    for m in a.models.split(","):
        train(m, a.epochs, dev)
