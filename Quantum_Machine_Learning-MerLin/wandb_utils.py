"""
Weights & Biases utility helpers for the QORC-MNIST demo.

All figure-creation and W&B logging functions live here so that the main
script stays focused on the ML/quantum workflow.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import wandb
import perceval as pcvl

from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix


# ---------------------------------------------------------------------------
# Core logging helper
# ---------------------------------------------------------------------------

def log_figure(run: "wandb.sdk.wandb_run.Run", key: str, fig: plt.Figure) -> None:
    """Log a Matplotlib figure to W&B and close it."""
    run.log({key: wandb.Image(fig)})
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure factories
# ---------------------------------------------------------------------------

def make_mnist_samples_figure(
    flat_images: np.ndarray,
    labels: np.ndarray,
    title: str,
    n_rows: int = 2,
    n_cols: int = 8,
) -> plt.Figure:
    """Grid of MNIST examples for pedagogical logging."""
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(1.5 * n_cols, 1.5 * n_rows))
    axes = axes.flatten()
    for idx, ax in enumerate(axes):
        if idx >= len(flat_images):
            ax.axis("off")
            continue
        ax.imshow(flat_images[idx].reshape(28, 28), cmap="gray")
        ax.set_title(f"label={int(labels[idx])}", fontsize=9)
        ax.axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    return fig


def make_pca_variance_figure(pca: PCA) -> plt.Figure:
    """PCA explained variance ratio bar + cumulative line."""
    comp_idx = np.arange(1, len(pca.explained_variance_ratio_) + 1)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(comp_idx, pca.explained_variance_ratio_, color="#2a9d8f")
    ax.plot(comp_idx, np.cumsum(pca.explained_variance_ratio_), marker="o", color="#e76f51")
    ax.set_xlabel("PCA component")
    ax.set_ylabel("Explained variance ratio")
    ax.set_title("PCA variance profile")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    return fig


def make_pca_projection_figure(X_pca: np.ndarray, y: np.ndarray, title: str) -> plt.Figure:
    """First two PCA dimensions colored by class."""
    fig, ax = plt.subplots(figsize=(6, 5))
    scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap="tab10", s=18, alpha=0.8)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title(title)
    legend = ax.legend(*scatter.legend_elements(), title="Digit", loc="best")
    ax.add_artist(legend)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    return fig


def make_pca_components_figure(pca: PCA, n_components: int) -> plt.Figure:
    """PCA component vectors visualized as 28×28 pseudo-images."""
    n_components = min(n_components, pca.components_.shape[0])
    n_cols = 4
    n_rows = int(np.ceil(n_components / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 3 * n_rows))
    axes = np.array(axes).reshape(-1)
    for idx, ax in enumerate(axes):
        if idx >= n_components:
            ax.axis("off")
            continue
        comp = pca.components_[idx].reshape(28, 28)
        im = ax.imshow(comp, cmap="coolwarm")
        ax.set_title(f"PC{idx + 1}")
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle("PCA components (reshaped to image space)")
    fig.tight_layout()
    return fig


def make_confusion_matrix_figure(y_true: np.ndarray, y_pred: np.ndarray, title: str) -> plt.Figure:
    """Confusion matrix figure from predictions."""
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(10))
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    ax.set_xticks(np.arange(10))
    ax.set_yticks(np.arange(10))
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Circuit export
# ---------------------------------------------------------------------------

def export_and_log_circuit(
    reservoir,
    run: "wandb.sdk.wandb_run.Run",
    output_dir: Path,
) -> None:
    """Export Perceval circuit visuals and log them to W&B."""
    output_dir.mkdir(parents=True, exist_ok=True)
    circuit_artifact = wandb.Artifact("perceval-circuit", type="visualization")

    circuit_text_path = output_dir / "reservoir_circuit.txt"
    circuit_text_path.write_text(str(reservoir.circuit), encoding="utf-8")
    circuit_artifact.add_file(str(circuit_text_path))

    if hasattr(pcvl, "pdisplay_to_file"):
        exported = False
        for ext in ("png", "svg", "pdf"):
            candidate = output_dir / f"reservoir_circuit.{ext}"
            try:
                pcvl.pdisplay_to_file(reservoir.circuit, str(candidate))
                if candidate.exists():
                    circuit_artifact.add_file(str(candidate))
                    if ext == "png":
                        run.log({"reservoir/circuit_diagram": wandb.Image(str(candidate))})
                    exported = True
                    break
            except Exception:
                continue
        if not exported:
            _log_circuit_text(run, reservoir)
    else:
        _log_circuit_text(run, reservoir)

    run.log_artifact(circuit_artifact)


def _log_circuit_text(run: "wandb.sdk.wandb_run.Run", reservoir) -> None:
    fig, ax = plt.subplots(figsize=(12, 2))
    ax.text(0.01, 0.5, str(reservoir.circuit), family="monospace", fontsize=9)
    ax.axis("off")
    fig.tight_layout()
    log_figure(run, "reservoir/circuit_text_render", fig)


# ---------------------------------------------------------------------------
# Feature CSV artifacts
# ---------------------------------------------------------------------------

def log_features_as_csv_artifact(
    run: "wandb.sdk.wandb_run.Run",
    artifact_name: str,
    output_dir: Path,
    arrays: dict[str, tuple[np.ndarray, np.ndarray]],
) -> None:
    """
    Save feature arrays as CSV files and upload them as a W&B artifact.

    Each entry in `arrays` maps a file stem to a (features, labels) pair.
    The resulting CSV has columns feature_0, feature_1, …, feature_N, label
    and is immediately downloadable from the W&B Artifacts browser.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact = wandb.Artifact(artifact_name, type="dataset",
                              description="Quantum reservoir feature embeddings (CSV)")

    for name, (features, labels) in arrays.items():
        n_features = features.shape[1]
        header = ",".join([f"feature_{i}" for i in range(n_features)] + ["label"])
        data = np.concatenate([features, labels.reshape(-1, 1)], axis=1)
        csv_path = output_dir / f"{name}.csv"
        np.savetxt(csv_path, data, delimiter=",", header=header, comments="")
        artifact.add_file(str(csv_path))

    run.log_artifact(artifact)


# ---------------------------------------------------------------------------
# Embedding artifact
# ---------------------------------------------------------------------------

def save_embeddings(
    output_dir: Path,
    run: "wandb.sdk.wandb_run.Run",
    embedding_dict: dict[str, np.ndarray],
) -> None:
    """Save embeddings/features as a W&B artifact."""
    output_dir.mkdir(parents=True, exist_ok=True)
    embedding_path = output_dir / "embeddings.npz"
    np.savez(embedding_path, **embedding_dict)
    artifact = wandb.Artifact("qorc-embeddings", type="dataset")
    artifact.add_file(str(embedding_path))
    run.log_artifact(artifact)
