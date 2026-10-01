"""
Create a 3x3 t-SNE figure:

Rows    : Overhead MNIST, SAT-6, So2Sat
Columns : Real/input features, S2Q-SEQNN-Shared, S2Q-SEQNN-Adaptive

The "Real" column uses actual normalized input samples flattened into vectors.
The model columns use the learned hybrid representation immediately before the
classifier. Missing model checkpoints are shown as empty panels so the script
can be run before both variants are trained.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.lines import Line2D
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

import s2q_tsne_visualize as tv


SCRIPT_DIR = Path(__file__).resolve().parent
DATASETS = ["overhead", "sat6", "so2sat"]
DATASET_TITLES = {
    "overhead": "Overhead MNIST",
    "sat6": "SAT-6",
    "so2sat": "So2Sat",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build 3x3 real/shared/adaptive t-SNE figure.")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--source", default="optimization", choices=["optimization", "basis", "gate"])
    parser.add_argument("--max-samples", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--perplexity", type=float, default=30.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dot-size", type=float, default=13.0)
    parser.add_argument("--fig-width", type=float, default=13.2)
    parser.add_argument("--fig-height", type=float, default=8.0)
    parser.add_argument("--legend-font-size", type=float, default=9.0)
    parser.add_argument("--legend-marker-size", type=float, default=5.8)
    parser.add_argument(
        "--adaptive-variant",
        default="legacy-adaptive",
        choices=["adaptive", "legacy-adaptive"],
        help="Use legacy-adaptive for your already completed Adaptive run.",
    )
    parser.add_argument(
        "--out-dir",
        default=str(SCRIPT_DIR / "s2q_tsne_figures" / "grid_3x3"),
    )
    return parser


def import_loader_module(dataset: str):
    result = {
        "overhead_mode": "seqnn5",
        "quick": False,
        "seed": 42,
        "frontend_mode": "auto",
    }
    return tv.import_model_module(
        SCRIPT_DIR / "s2q_seqnn_adaptive.py",
        dataset,
        result,
        f"s2q_grid_loader_{dataset}",
    )


def select_split(module, dataset: str, split: str):
    tr_x, tr_y, val_x, val_y, te_x, te_y = module.load_data(dataset)
    if split == "train":
        return tr_x, tr_y
    if split == "val":
        return val_x, val_y
    return te_x, te_y


def run_tsne(features: np.ndarray, labels: np.ndarray, args) -> np.ndarray:
    scaled = StandardScaler().fit_transform(features)
    perplexity = min(args.perplexity, max(5.0, (len(labels) - 1) / 3.0))
    return TSNE(
        n_components=2,
        perplexity=perplexity,
        init="pca",
        learning_rate="auto",
        random_state=args.seed,
    ).fit_transform(scaled)


def raw_panel(dataset: str, args) -> dict:
    module = import_loader_module(dataset)
    X, y = select_split(module, dataset, args.split)
    X, y = tv.stratified_cap(X, y, args.max_samples, args.seed)
    features = X.reshape(len(X), -1)
    coords = run_tsne(features, y, args)
    return {
        "coords": coords,
        "labels": y,
        "class_names": module.META[dataset]["classes"],
        "meta": {
            "variant": "real",
            "dataset": dataset,
            "split": args.split,
            "samples": int(len(y)),
            "feature_dim": int(features.shape[1]),
        },
    }


def model_panel(variant: str, dataset: str, args) -> dict:
    defaults = tv.variant_defaults(variant)
    summary_root = Path(defaults["summary_root"])
    outputs_root = Path(defaults["outputs_root"])
    model_script = Path(defaults["model_script"])

    summary_folder = tv.find_summary_folder(summary_root, dataset, args.source)
    best_run, result_folder_name = tv.choose_best_run(summary_folder, dataset, args.source)
    result_folder = outputs_root / result_folder_name
    result_json = result_folder / f"results_{dataset}.json"
    model_path = result_folder / f"sq_seqnn_{dataset}.pt"
    if not result_json.exists() or not model_path.exists():
        raise FileNotFoundError(f"Missing {result_json} or {model_path}")

    with open(result_json) as f:
        result = json.load(f)

    module = tv.import_model_module(
        model_script,
        dataset,
        result,
        f"s2q_grid_{variant}_{dataset}_{abs(hash(str(result_folder))) % 10**9}",
    )
    X, y = select_split(module, dataset, args.split)
    X, y = tv.stratified_cap(X, y, args.max_samples, args.seed)

    model = module.SQ_SEQNN(
        module.META[dataset]["C"],
        module.META[dataset]["size"],
        module.META[dataset]["K"],
    ).to(module.DEVICE)
    state = torch.load(model_path, map_location=module.DEVICE)
    model.load_state_dict(state)
    features, labels = tv.extract_hybrid_features(module, model, X, y, args.batch_size)
    coords = run_tsne(features, labels, args)
    return {
        "coords": coords,
        "labels": labels,
        "class_names": module.META[dataset]["classes"],
        "meta": {
            "variant": variant,
            "dataset": dataset,
            "split": args.split,
            "samples": int(len(labels)),
            "feature_dim": int(features.shape[1]),
            "summary_folder": str(summary_folder),
            "result_folder": str(result_folder),
            "model_path": str(model_path),
            "run_tag": str(best_run.get("run_tag", "")),
            "accuracy": result.get("accuracy"),
            "balanced_accuracy": result.get("balanced_accuracy"),
            "total_params": result.get("total_params"),
        },
    }


def save_panel_csv(panel: dict, out_dir: Path, dataset: str, col_key: str) -> None:
    coords = panel["coords"]
    labels = panel["labels"]
    names = panel["class_names"]
    df = pd.DataFrame({
        "x": coords[:, 0],
        "y": coords[:, 1],
        "label": labels,
        "class_name": [names[int(i)] for i in labels],
        "dataset": dataset,
        "panel": col_key,
    })
    for key, value in panel["meta"].items():
        if isinstance(value, (str, int, float, bool)) or value is None:
            df[key] = value
    df.to_csv(out_dir / f"tsne_grid_{dataset}_{col_key}.csv", index=False)
    with open(out_dir / f"tsne_grid_{dataset}_{col_key}.json", "w") as f:
        json.dump(panel["meta"], f, indent=2)


def plot_panel(ax, panel: dict, title: str, dot_size: float) -> None:
    coords = panel["coords"]
    labels = panel["labels"]
    names = panel["class_names"]
    cmap = plt.get_cmap("tab10")
    for class_idx, class_name in enumerate(names):
        mask = labels == class_idx
        if mask.any():
            ax.scatter(
                coords[mask, 0],
                coords[mask, 1],
                s=dot_size,
                alpha=0.82,
                color=cmap(class_idx % 10),
                label=class_name,
                edgecolors="none",
            )
    if title:
        ax.set_title(title, fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(alpha=0.12)
    for spine in ax.spines.values():
        spine.set_linewidth(0.65)


def plot_missing(ax, title: str, message: str) -> None:
    if title:
        ax.set_title(title, fontsize=10)
    ax.text(
        0.5,
        0.5,
        message,
        ha="center",
        va="center",
        fontsize=9,
        wrap=True,
        transform=ax.transAxes,
    )
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(alpha=0.12)
    for spine in ax.spines.values():
        spine.set_linewidth(0.65)


def add_row_legend(
    ax,
    class_names: list[str],
    font_size: float,
    marker_size: float,
) -> None:
    ax.axis("off")
    if not class_names:
        return
    cmap = plt.get_cmap("tab10")
    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="None",
            markersize=marker_size,
            markerfacecolor=cmap(i % 10),
            markeredgecolor="none",
            label=name,
        )
        for i, name in enumerate(class_names)
    ]
    ax.legend(
        handles=handles,
        loc="center",
        ncol=min(len(class_names), 6),
        frameon=False,
        fontsize=font_size,
        handletextpad=0.28,
        columnspacing=0.72,
        borderaxespad=0.0,
    )


def main() -> None:
    args = build_parser().parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    columns = [
        ("real", "Real/Input"),
        ("shared", "S²Q-SEQNN-Shared"),
        (args.adaptive_variant, "S²Q-SEQNN-Adaptive"),
    ]
    fig = plt.figure(figsize=(args.fig_width, args.fig_height))
    grid = fig.add_gridspec(
        6,
        3,
        height_ratios=[1.0, 0.072, 1.0, 0.072, 1.0, 0.078],
        hspace=0.055,
        wspace=0.025,
    )
    axes = np.empty((3, 3), dtype=object)
    legend_axes = []
    for row_idx in range(3):
        plot_row = row_idx * 2
        for col_idx in range(3):
            axes[row_idx, col_idx] = fig.add_subplot(grid[plot_row, col_idx])
        legend_axes.append(fig.add_subplot(grid[plot_row + 1, :]))

    for row_idx, dataset in enumerate(DATASETS):
        row_class_names = None
        for col_idx, (col_key, col_title) in enumerate(columns):
            ax = axes[row_idx, col_idx]
            try:
                if col_key == "real":
                    panel = raw_panel(dataset, args)
                else:
                    panel = model_panel(col_key, dataset, args)
                save_panel_csv(panel, out_dir, dataset, col_key)
                row_class_names = row_class_names or panel["class_names"]
                plot_panel(ax, panel, "", dot_size=args.dot_size)
            except Exception as exc:
                plot_missing(ax, "", f"Not available yet\n{type(exc).__name__}")
                with open(out_dir / f"tsne_grid_{dataset}_{col_key}_missing.txt", "w") as f:
                    f.write(str(exc))
                    f.write("\n")
            if row_idx == 0:
                ax.set_title(col_title, fontsize=10.5, fontweight="bold", pad=6)
            if col_idx == 0:
                ax.set_ylabel(
                    DATASET_TITLES[dataset],
                    fontsize=10.5,
                    fontweight="bold",
                    labelpad=10,
                )
        add_row_legend(
            legend_axes[row_idx],
            row_class_names or [],
            font_size=args.legend_font_size,
            marker_size=args.legend_marker_size,
        )

    fig.subplots_adjust(left=0.052, right=0.995, top=0.955, bottom=0.035)
    fig_path = out_dir / f"tsne_3x3_real_shared_adaptive_{args.split}_{args.source}.png"
    pdf_path = out_dir / f"tsne_3x3_real_shared_adaptive_{args.split}_{args.source}.pdf"
    plt.savefig(fig_path, dpi=500)
    plt.savefig(pdf_path)
    plt.close()
    print(f"Saved 3x3 t-SNE figure: {fig_path}")
    print(f"Saved 3x3 t-SNE PDF:    {pdf_path}")


if __name__ == "__main__":
    main()
