from __future__ import annotations

import math
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def _as_image(x: np.ndarray) -> np.ndarray:
    return x.reshape(28, 28)


def plot_digit_grid(X: np.ndarray, rows: int = 10, cols: int = 10):
    n = min(len(X), rows * cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.2, rows * 1.2))
    axes = axes.ravel()
    for i in range(rows * cols):
        axes[i].axis("off")
        if i < n:
            axes[i].imshow(_as_image(X[i]), cmap="binary")
    fig.tight_layout(pad=0.1)
    return fig


def plot_class_distribution(y: np.ndarray):
    labels, counts = np.unique(y, return_counts=True)
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.barplot(x=labels, y=counts, ax=ax, palette="viridis")
    ax.set_xlabel("Class")
    ax.set_ylabel("Count")
    ax.set_title("Class distribution")
    return fig


def plot_pca_scatter(X: np.ndarray, y: np.ndarray):
    pca = PCA(n_components=2, random_state=42)
    X2 = pca.fit_transform(X)
    fig, ax = plt.subplots(figsize=(6, 5))
    scatter = ax.scatter(X2[:, 0], X2[:, 1], c=y.astype(int), cmap="tab10", s=10, alpha=0.7)
    ax.set_title("PCA (2D)")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    handles, _ = scatter.legend_elements(num=10)
    ax.legend(handles, [str(i) for i in range(10)], title="Class", loc="upper right", fontsize=8)
    return fig


def plot_tsne_scatter(X: np.ndarray, y: np.ndarray, perplexity: int = 30):
    tsne = TSNE(n_components=2, learning_rate="auto", perplexity=perplexity, init="pca", random_state=42)
    X2 = tsne.fit_transform(X)
    fig, ax = plt.subplots(figsize=(6, 5))
    scatter = ax.scatter(X2[:, 0], X2[:, 1], c=y.astype(int), cmap="tab10", s=8, alpha=0.7)
    ax.set_title(f"t-SNE (2D), perplexity={perplexity}")
    ax.set_xlabel("Dim 1")
    ax.set_ylabel("Dim 2")
    handles, _ = scatter.legend_elements(num=10)
    ax.legend(handles, [str(i) for i in range(10)], title="Class", loc="upper right", fontsize=8)
    return fig


