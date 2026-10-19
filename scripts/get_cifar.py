"""Fetch CIFAR-10 (Hugging Face mirror of the Toronto release) into
data/cifar10.npz as uint8 NCHW arrays. Needs pyarrow and pillow."""
import io
import os
import urllib.request

import numpy as np
import pyarrow.parquet as pq
from PIL import Image

BASE = "https://huggingface.co/datasets/uoft-cs/cifar10/resolve/main/plain_text/"
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "cifar10.npz")


def split(fname):
    raw = urllib.request.urlopen(BASE + fname).read()
    t = pq.read_table(io.BytesIO(raw)).to_pydict()
    x = np.stack([np.asarray(Image.open(io.BytesIO(im["bytes"])).convert("RGB")) for im in t["img"]])
    return x.transpose(0, 3, 1, 2).copy(), np.array(t["label"], dtype=np.int64)


if __name__ == "__main__":
    xtr, ytr = split("train-00000-of-00001.parquet")
    xte, yte = split("test-00000-of-00001.parquet")
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    np.savez(OUT, x_train=xtr, y_train=ytr, x_test=xte, y_test=yte)
    print(xtr.shape, xte.shape, np.bincount(yte))
