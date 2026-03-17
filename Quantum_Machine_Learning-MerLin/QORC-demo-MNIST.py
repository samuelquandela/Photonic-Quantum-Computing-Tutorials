#!/usr/bin/env python3
"""
QORC Demo on MNIST with Weights & Biases logging.

Benchmarks a Quantum Optical Reservoir Computing (QORC) workflow on a small
stratified MNIST subset and logs artifacts to W&B.
"""

from __future__ import annotations

import random
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import wandb
import merlin as ML
import perceval as pcvl

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.svm import LinearSVC
from torchvision.datasets import MNIST

from token_utils import load_cloud_token
from wandb_utils import (
    log_figure,
    make_mnist_samples_figure,
    make_pca_variance_figure,
    make_pca_projection_figure,
    make_pca_components_figure,
    make_confusion_matrix_figure,
    export_and_log_circuit,
    log_features_as_csv_artifact,
    save_embeddings,
)


# ---------------------------------------------------------------------------
# Variables and hyperparameters
# ---------------------------------------------------------------------------
PER_CLASS_TRAIN = 30
PER_CLASS_TEST = 100
N_QFEATURES = 12
VAL_FRACTION = 0.2
READOUT_EPOCHS = 300
READOUT_LR = 1e-2
CONFUSION_LOG_EVERY = 25
WANDB_PROJECT = "webinar-test"


def reset_seeds(seed: int = 123) -> None:
    """Fix all random seeds so results are fully reproducible."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def stratified_subset(
    X: torch.Tensor, y: torch.Tensor, per_class: int, seed: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return a balanced subset with exactly `per_class` samples per class."""
    g = torch.Generator().manual_seed(seed)
    indices = []
    for cls in range(10):
        cls_idx = torch.where(y == cls)[0]
        perm = cls_idx[torch.randperm(len(cls_idx), generator=g)]
        indices.append(perm[:per_class])
    selected = torch.cat(indices)
    selected = selected[torch.randperm(len(selected), generator=g)]
    return X[selected], y[selected]


def minmax_normalize(data: np.ndarray, global_min: float, global_max: float) -> np.ndarray:
    """Scale `data` to [0, 1] using pre-computed global min and max."""
    epsilon = 1e-8
    return (data - global_min) / (global_max - global_min + epsilon)


def standard_normalize(data: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """Zero-centre and unit-scale `data` with pre-computed train statistics."""
    epsilon = 1e-8
    return (data - mean) / (std + epsilon)


def as_numpy(x) -> np.ndarray:
    """Convert tensor or array-like to NumPy."""
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def train_linear_readout(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    run: "wandb.sdk.wandb_run.Run",
    run_prefix: str,
    n_epochs: int = READOUT_EPOCHS,
    lr: float = READOUT_LR,
    seed: int = 123,
    confusion_every: int = CONFUSION_LOG_EVERY,
) -> tuple[np.ndarray, torch.nn.Linear]:
    """Train a linear readout and log curves/confusion matrices to W&B."""
    torch.manual_seed(seed)

    n_features = X_train.shape[1]
    model = torch.nn.Linear(n_features, 10)
    optimiser = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.CrossEntropyLoss()

    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.long)
    X_val_t = torch.tensor(X_val, dtype=torch.float32)
    y_val_t = torch.tensor(y_val, dtype=torch.long)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    y_test_t = torch.tensor(y_test, dtype=torch.long)

    run.summary[f"{run_prefix}_model"] = str(model)
    try:
        run.watch(model, log="all", log_freq=max(1, n_epochs // 10))
    except Exception:
        pass

    for epoch in range(1, n_epochs + 1):
        model.train()
        optimiser.zero_grad()
        train_logits = model(X_train_t)
        train_loss = loss_fn(train_logits, y_train_t)
        train_loss.backward()
        optimiser.step()

        model.eval()
        with torch.no_grad():
            val_logits = model(X_val_t)
            val_loss = loss_fn(val_logits, y_val_t)

            test_logits = model(X_test_t)
            train_preds = train_logits.argmax(dim=1).cpu().numpy()
            val_preds = val_logits.argmax(dim=1).cpu().numpy()
            test_preds = test_logits.argmax(dim=1).cpu().numpy()

        train_acc = accuracy_score(y_train, train_preds)
        val_acc = accuracy_score(y_val, val_preds)
        test_acc = accuracy_score(y_test, test_preds)

        run.log(
            {
                f"{run_prefix}/train_loss": float(train_loss.item()),
                f"{run_prefix}/val_loss": float(val_loss.item()),
                f"{run_prefix}/train_acc": float(train_acc),
                f"{run_prefix}/val_acc": float(val_acc),
                f"{run_prefix}/test_acc": float(test_acc),
                f"{run_prefix}/epoch": epoch,
            },
            step=epoch,
        )

        if epoch % confusion_every == 0 or epoch == 1 or epoch == n_epochs:
            cm_fig = make_confusion_matrix_figure(
                y_true=y_test,
                y_pred=test_preds,
                title=f"{run_prefix} - test confusion matrix (epoch {epoch})",
            )
            run.log({f"{run_prefix}/test_confusion_matrix": wandb.Image(cm_fig)}, step=epoch)
            plt.close(cm_fig)

    model.eval()
    with torch.no_grad():
        final_logits = model(X_test_t)
        final_preds = final_logits.argmax(dim=1).cpu().numpy()

    run.summary[f"{run_prefix}_final_test_acc"] = float(accuracy_score(y_test, final_preds))
    return final_preds, model


def main() -> None:
    reset_seeds(123)

    script_path = Path(__file__).resolve()
    project_root = script_path.parent.parent
    output_dir = script_path.parent / "wandb_outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    run = wandb.init(
        project=WANDB_PROJECT,
        name=f"qorc-mnist-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
        config={
            "per_class_train": PER_CLASS_TRAIN,
            "per_class_test": PER_CLASS_TEST,
            "n_qfeatures": N_QFEATURES,
            "val_fraction": VAL_FRACTION,
            "readout_epochs": READOUT_EPOCHS,
            "readout_lr": READOUT_LR,
            "confusion_log_every": CONFUSION_LOG_EVERY,
            "seed": 123,
        },
    )

    try:
        run.log_code(
            root=str(project_root),
            include_fn=lambda p: p.endswith((".py", ".ipynb", ".md", ".txt", ".yml", ".yaml")),
        )

        # 1) Load MNIST and build stratified subsets.
        data_root = Path("./data")
        train_ds = MNIST(root=data_root, train=True, download=True)
        test_ds = MNIST(root=data_root, train=False, download=True)

        X_train_full = train_ds.data.float().view(-1, 28 * 28) / 255.0
        y_train_full = train_ds.targets.long()
        X_test_full = test_ds.data.float().view(-1, 28 * 28) / 255.0
        y_test_full = test_ds.targets.long()

        X_train, y_train = stratified_subset(
            X_train_full, y_train_full, per_class=PER_CLASS_TRAIN, seed=123
        )
        X_test, y_test = stratified_subset(
            X_test_full, y_test_full, per_class=PER_CLASS_TEST, seed=456
        )

        print(f"Training samples : {X_train.shape[0]}")
        print(f"Test samples     : {X_test.shape[0]}")
        print(f"Input dimension  : {X_train.shape[1]}")

        X_train_pixels = X_train.numpy()
        y_train_np = y_train.numpy()
        X_test_pixels = X_test.numpy()
        y_test_np = y_test.numpy()

        # Visualize input images.
        image_fig = make_mnist_samples_figure(
            X_train_pixels[:16],
            y_train_np[:16],
            title="MNIST training sample preview",
        )
        log_figure(run, "data/sample_images", image_fig)

        # Save selected images as CSV artifacts (same format as feature CSVs).
        log_features_as_csv_artifact(
            run,
            artifact_name="mnist-selected-images",
            output_dir=output_dir / "images",
            arrays={
                "X_train_pixels": (X_train_pixels, y_train_np),
                "X_test_pixels": (X_test_pixels, y_test_np),
            },
        )

        # 2) PCA + min-max scaling.
        pca = PCA(n_components=N_QFEATURES, random_state=123)
        X_train_pca = pca.fit_transform(X_train_pixels)
        X_test_pca = pca.transform(X_test_pixels)

        pca_global_min = X_train_pca.min()
        pca_global_max = X_train_pca.max()

        X_train_q = minmax_normalize(X_train_pca, pca_global_min, pca_global_max)
        X_test_q = minmax_normalize(X_test_pca, pca_global_min, pca_global_max)

        print(f"\nQuantum input size : {X_train_q.shape[1]}")
        print(f"PCA global min/max : {pca_global_min:.4f} / {pca_global_max:.4f}")

        run.log(
            {
                "pca/global_min": float(pca_global_min),
                "pca/global_max": float(pca_global_max),
                "pca/explained_variance_sum": float(pca.explained_variance_ratio_.sum()),
            }
        )

        log_figure(run, "pca/explained_variance", make_pca_variance_figure(pca))
        log_figure(
            run,
            "pca/train_projection_pc1_pc2",
            make_pca_projection_figure(X_train_pca, y_train_np, "Train PCA projection"),
        )
        log_figure(
            run,
            "pca/components",
            make_pca_components_figure(pca, n_components=N_QFEATURES),
        )

        # 3) Baseline (LinearSVC on raw pixels).
        baseline = LinearSVC(dual=False, random_state=123, max_iter=20_000)
        baseline.fit(X_train_pixels, y_train_np)
        baseline_preds = baseline.predict(X_test_pixels)
        baseline_acc = accuracy_score(y_test_np, baseline_preds)

        print(f"\nL-SVC baseline accuracy : {baseline_acc:.4f}")
        run.log({"baseline/accuracy": float(baseline_acc)})
        log_figure(
            run,
            "baseline/test_confusion_matrix",
            make_confusion_matrix_figure(
                y_test_np,
                baseline_preds,
                "Baseline L-SVC - test confusion matrix",
            ),
        )

        # 4) Build QORC reservoir.
        builder = ML.CircuitBuilder(n_modes=12)
        builder.add_entangling_layer()
        builder.add_angle_encoding(modes=range(N_QFEATURES))
        builder.add_entangling_layer()

        reservoir = ML.QuantumLayer(
            builder=builder,
            input_state=[1, 0, 1, 0, 1, 0] + ([0] * (N_QFEATURES - 6)),
        ).eval()

        # Keep existing terminal rendering behavior and add W&B export.
        pcvl.pdisplay(reservoir.circuit)
        export_and_log_circuit(reservoir, run, output_dir / "circuit")

        def extract_ideal_features(x_np: np.ndarray) -> np.ndarray:
            with torch.no_grad():
                z = reservoir(torch.tensor(x_np, dtype=torch.float32))
            return as_numpy(z)

        # 5) MerLin ideal run.
        Z_train_ideal = extract_ideal_features(X_train_q)
        Z_test_ideal = extract_ideal_features(X_test_q)

        log_features_as_csv_artifact(
            run,
            artifact_name="qorc-ideal-features",
            output_dir=output_dir / "features",
            arrays={
                "Z_train_ideal": (Z_train_ideal, y_train_np),
                "Z_test_ideal": (Z_test_ideal, y_test_np),
            },
        )

        qorc_mean_ideal = Z_train_ideal.mean(axis=0)
        qorc_std_ideal = Z_train_ideal.std(axis=0)

        Z_train_ideal_norm = standard_normalize(Z_train_ideal, qorc_mean_ideal, qorc_std_ideal)
        Z_test_ideal_norm = standard_normalize(Z_test_ideal, qorc_mean_ideal, qorc_std_ideal)

        X_train_ideal = np.concatenate([X_train_pixels, Z_train_ideal_norm], axis=1)
        X_test_ideal = np.concatenate([X_test_pixels, Z_test_ideal_norm], axis=1)

        X_train_ideal_fit, X_val_ideal, y_train_ideal_fit, y_val_ideal = train_test_split(
            X_train_ideal,
            y_train_np,
            test_size=VAL_FRACTION,
            random_state=123,
            stratify=y_train_np,
        )

        ideal_preds, ideal_model = train_linear_readout(
            X_train=X_train_ideal_fit,
            y_train=y_train_ideal_fit,
            X_val=X_val_ideal,
            y_val=y_val_ideal,
            X_test=X_test_ideal,
            y_test=y_test_np,
            run=run,
            run_prefix="ideal_readout",
            seed=123,
        )
        ideal_acc = accuracy_score(y_test_np, ideal_preds)
        print(f"MerLin ideal accuracy   : {ideal_acc:.4f}")

        run.log({"ideal/accuracy": float(ideal_acc)})
        run.summary["ideal_model_parameters"] = int(
            sum(p.numel() for p in ideal_model.parameters())
        )

        # 6) MerLinProcessor run (optional, requires CLOUD_TOKEN).
        CLOUD_TOKEN = load_cloud_token()
        processor_acc = None

        embeddings_payload: dict[str, np.ndarray] = {
            "X_train_pixels": X_train_pixels,
            "X_test_pixels": X_test_pixels,
            "X_train_pca": X_train_pca,
            "X_test_pca": X_test_pca,
            "X_train_q": X_train_q,
            "X_test_q": X_test_q,
            "Z_train_ideal": Z_train_ideal,
            "Z_test_ideal": Z_test_ideal,
            "Z_train_ideal_norm": Z_train_ideal_norm,
            "Z_test_ideal_norm": Z_test_ideal_norm,
            "y_train": y_train_np,
            "y_test": y_test_np,
        }

        if not CLOUD_TOKEN:
            print("\nMerLinProcessor run skipped – set CLOUD_TOKEN in your .env file.")
            run.summary["processor_status"] = "skipped_no_cloud_token"
        else:
            print("\nConnecting to remote quantum processor (sim:belenos)…")
            pcvl.RemoteConfig.set_token(CLOUD_TOKEN)
            remote_processor = pcvl.RemoteProcessor("sim:belenos")
            print("Connection established.")

            proc = ML.MerlinProcessor(
                remote_processor,
                microbatch_size=32,
                timeout=3600.0,
                max_shots_per_call=1000,
                chunk_concurrency=1,
            )

            x_train_q_t = torch.tensor(X_train_q, dtype=torch.float32)
            x_test_q_t = torch.tensor(X_test_q, dtype=torch.float32)

            print(f"\nSending training data to the quantum processor ({X_train_q.shape[0]} samples)…")
            Z_train_proc = as_numpy(proc.forward(reservoir, x_train_q_t))
            print("Training data retrieved.")

            print(f"Sending test data to the quantum processor ({X_test_q.shape[0]} samples)…")
            Z_test_proc = as_numpy(proc.forward(reservoir, x_test_q_t))
            print("Test data retrieved.")

            log_features_as_csv_artifact(
                run,
                artifact_name="qorc-processor-features",
                output_dir=output_dir / "features",
                arrays={
                    "Z_train_proc": (Z_train_proc, y_train_np),
                    "Z_test_proc": (Z_test_proc, y_test_np),
                },
            )

            print("\nNormalising quantum features…")
            qorc_mean_proc = Z_train_proc.mean(axis=0)
            qorc_std_proc = Z_train_proc.std(axis=0)

            Z_train_proc_norm = standard_normalize(Z_train_proc, qorc_mean_proc, qorc_std_proc)
            Z_test_proc_norm = standard_normalize(Z_test_proc, qorc_mean_proc, qorc_std_proc)

            X_train_proc = np.concatenate([X_train_pixels, Z_train_proc_norm], axis=1)
            X_test_proc = np.concatenate([X_test_pixels, Z_test_proc_norm], axis=1)

            X_train_proc_fit, X_val_proc, y_train_proc_fit, y_val_proc = train_test_split(
                X_train_proc,
                y_train_np,
                test_size=VAL_FRACTION,
                random_state=123,
                stratify=y_train_np,
            )

            print("Training linear readout on processor features…")
            proc_preds, proc_model = train_linear_readout(
                X_train=X_train_proc_fit,
                y_train=y_train_proc_fit,
                X_val=X_val_proc,
                y_val=y_val_proc,
                X_test=X_test_proc,
                y_test=y_test_np,
                run=run,
                run_prefix="processor_readout",
                seed=123,
            )
            processor_acc = accuracy_score(y_test_np, proc_preds)
            print(f"MerLinProcessor accuracy: {processor_acc:.4f}")
            run.log({"processor/accuracy": float(processor_acc)})
            run.summary["processor_model_parameters"] = int(
                sum(p.numel() for p in proc_model.parameters())
            )

            embeddings_payload.update(
                {
                    "Z_train_processor": Z_train_proc,
                    "Z_test_processor": Z_test_proc,
                    "Z_train_processor_norm": Z_train_proc_norm,
                    "Z_test_processor_norm": Z_test_proc_norm,
                }
            )

        # 7) Save embeddings and final result table.
        save_embeddings(output_dir=output_dir, run=run, embedding_dict=embeddings_payload)

        rows = [
            ["L-SVC baseline", float(baseline_acc)],
            ["MerLin ideal (local sim)", float(ideal_acc)],
            [
                "MerLinProcessor (remote)",
                None if processor_acc is None else float(processor_acc),
            ],
        ]
        results_table = wandb.Table(columns=["run", "accuracy"], data=rows)
        run.log({"results/accuracy_table": results_table})

        print("\n" + "=" * 55)
        print("Accuracy comparison")
        print("=" * 55)
        print(f"  L-SVC baseline              : {baseline_acc:.4f}")
        print(f"  MerLin ideal (local sim)    : {ideal_acc:.4f}")
        if processor_acc is None:
            print("  MerLinProcessor             : skipped (no CLOUD_TOKEN)")
        else:
            print(f"  MerLinProcessor (hardware)  : {processor_acc:.4f}")
        print("=" * 55)

    finally:
        wandb.finish()


if __name__ == "__main__":
    main()
