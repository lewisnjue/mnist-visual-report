import os
from pathlib import Path

import streamlit as st

from src.data import load_mnist_cached, get_class_distribution, get_sample_subset
from src.plots import (
    plot_digit_grid,
    plot_class_distribution,
    plot_pca_scatter,
    plot_tsne_scatter,
)


st.set_page_config(
    page_title="MNIST Classification Visual Report",
    page_icon="🧮",
    layout="wide",
)


@st.cache_data(show_spinner=False)
def _load_data():
    X, y, descr = load_mnist_cached()
    return X, y, descr


def section_header(title: str, subtitle: str | None = None):
    st.markdown(f"## {title}")
    if subtitle:
        st.caption(subtitle)


def main():
    st.title("MNIST Classification Visual Report")
    st.write(
        "Interactive visualizations and a concise report summarizing key concepts in classification using the MNIST dataset."
    )

    with st.spinner("Loading MNIST dataset (cached after first run)..."):
        X, y, descr = _load_data()

    # Sidebar controls
    st.sidebar.header("Controls")
    sample_count = st.sidebar.slider("Sample size for visuals", 200, 5000, 1000, step=200)
    show_tsne = st.sidebar.checkbox("Compute t-SNE (slower)", value=False)
    tsne_perplexity = st.sidebar.slider("t-SNE perplexity", 5, 50, 30, step=5)

    Xs, ys = get_sample_subset(X, y, n_samples=sample_count, seed=42)

    # Problem statement
    section_header("Problem statement")
    st.write(
        "Given 28×28 grayscale images of handwritten digits (0–9), the goal is to classify each image into the correct digit class. We explore dataset characteristics and core evaluation techniques (precision/recall, ROC, confusion matrix) with emphasis on visualization rather than heavy model training."
    )

    # Dataset overview
    section_header("Dataset overview", "MNIST: 70,000 images, 28×28 pixels, 10 classes")
    col1, col2 = st.columns([2, 1])
    with col1:
        fig_grid = plot_digit_grid(Xs, rows=10, cols=10)
        st.pyplot(fig_grid, use_container_width=True)
        st.caption("Random sample of digits.")
    with col2:
        fig_dist = plot_class_distribution(ys)
        st.pyplot(fig_dist, use_container_width=True)
        st.caption("Class distribution in the sampled subset.")

    with st.expander("Dataset description (from OpenML)"):
        st.text(descr[:4000] + ("..." if len(descr) > 4000 else ""))

    # Methods & architecture (conceptual)
    section_header("Methods & architecture")
    st.markdown(
        "- We demonstrate evaluation concepts using lightweight computations.\n"
        "- Visualizations include class balance, dimensionality reduction (PCA, optional t-SNE).\n"
        "- Heavy training is intentionally omitted by default to keep the app responsive."
    )

    # Results (visuals)
    section_header("Results (metrics, visuals)")
    c1, c2 = st.columns(2)
    with c1:
        fig_pca = plot_pca_scatter(Xs, ys)
        st.pyplot(fig_pca, use_container_width=True)
        st.caption("PCA (2D) embedding colored by class.")
    with c2:
        if show_tsne:
            with st.spinner("Computing t-SNE (can take ~10–60s depending on sample size)..."):
                fig_tsne = plot_tsne_scatter(Xs, ys, perplexity=tsne_perplexity)
            st.pyplot(fig_tsne, use_container_width=True)
            st.caption("t-SNE (2D) embedding colored by class.")
        else:
            st.info("Enable t-SNE in the sidebar to compute and visualize non-linear embeddings.")


    # Further improvements
    section_header("Further improvements")
    st.markdown(
        "- Add confusion matrix and precision–recall visuals based on a fast baseline (e.g., LogisticRegression on a small subset).\n"
        "- Try simple augmentation (shifts/rotations) to observe effects on embeddings.\n"
        "- Compare embeddings across different subsets (e.g., only 3 vs 5).\n"
        "- Persist precomputed embeddings to speed up initial load on cloud."
    )

    # Footer
    st.write("\n")
    st.caption("Built with Streamlit. Data: MNIST via OpenML. © lewisnjue")


if __name__ == "__main__":
    # Ensure assets directory exists at runtime (for local use)
    Path("assets/figures").mkdir(parents=True, exist_ok=True)
    main()


