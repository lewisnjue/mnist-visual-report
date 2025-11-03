from pathlib import Path

import matplotlib.pyplot as plt
from src.data import load_mnist_cached, get_sample_subset
from src.plots import plot_digit_grid


def main():
    X, y, _ = load_mnist_cached()
    Xs, ys = get_sample_subset(X, y, n_samples=100, seed=7)
    fig = plot_digit_grid(Xs, rows=10, cols=10)
    out_dir = Path("assets/figures")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "sample_grid.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()


