from __future__ import annotations

from typing import Tuple

import numpy as np
from sklearn.datasets import fetch_openml


def load_mnist_cached() -> Tuple[np.ndarray, np.ndarray, str]:
    mnist = fetch_openml("mnist_784", as_frame=False)
    X, y = mnist.data, mnist.target
    descr = mnist.DESCR if hasattr(mnist, "DESCR") else "MNIST dataset"
    return X, y, descr


def get_sample_subset(
    X: np.ndarray, y: np.ndarray, n_samples: int = 1000, seed: int | None = 42
) -> Tuple[np.ndarray, np.ndarray]:
    n_samples = int(min(max(1, n_samples), len(X)))
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X), size=n_samples, replace=False)
    return X[idx], y[idx]


def get_class_distribution(y: np.ndarray) -> dict[str, int]:
    unique, counts = np.unique(y, return_counts=True)
    return {str(u): int(c) for u, c in zip(unique, counts)}


