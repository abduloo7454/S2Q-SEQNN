"""
==============================================================================
SQ-SEQNN: Scattering-Quantum SEQNN
Standalone low-parameter hybrid model with fixed multi-scale scattering-style
features, IQP re-uploading QCNN, and a lightweight skip-connected classifier.

Examples
--------
Quick smoke test on Overhead-MNIST full 10-class setup:
    python3 sq_seqnn_fast.py --dataset overhead --quick

SEQNN-style 5-class Overhead-MNIST setup:
    python3 sq_seqnn_fast.py --dataset overhead --overhead-mode seqnn5 --quick

Full training:
    python3 sq_seqnn_fast.py --dataset overhead
==============================================================================
"""

import argparse
import json
import math
import os
import random
import sys
import time
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

try:
    import pennylane as qml
except Exception as exc:
    msg = str(exc)
    if "NumpyMimic" in msg or "autoray" in msg:
        sys.stderr.write(
            "\nPennyLane failed to import because this script is using the wrong Python interpreter.\n"
            "Your current interpreter is loading user-site packages instead of the project virtual environment.\n\n"
            "Use this command instead:\n"
            "  MPLCONFIGDIR=\"$PWD/.mplconfig\" python src/sq_seqnn_fast.py --dataset overhead\n\n"
            "Or activate the virtual environment first:\n"
            "  source .venv/bin/activate\n\n"
        )
    raise

warnings.filterwarnings("ignore")


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        default="overhead",
        choices=["overhead", "sat6", "so2sat"],
    )
    parser.add_argument(
        "--overhead-mode",
        default="seqnn5",
        choices=["full10", "fair5", "seqnn5"],
        help="Use all 10 Overhead classes or the exact 5-class SEQNN-style split.",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Fast smoke test with reduced samples and epochs.",
    )
    parser.add_argument("--n-qubits", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--cls-hidden", type=int, default=None)
    parser.add_argument("--lr-q-factor", type=float, default=None)
    parser.add_argument("--label-smooth", type=float, default=None)
    parser.add_argument("--n-reupload", type=int, default=None,
                        help="Data re-uploading rounds R (default 2, the paper setting).")
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--model-path", type=str, default="")
    parser.add_argument(
        "--tta",
        default="auto",
        choices=["auto", "none", "flip", "rot4", "d4"],
        help="Test-time augmentation used during evaluation.",
    )
    parser.add_argument(
        "--measure-bases",
        default="auto",
        choices=["auto", "x", "y", "z", "xz", "xyz"],
        help="Quantum measurement basis for the readout layer.",
    )
    parser.add_argument(
        "--gate-type",
        default="auto",
        choices=["auto", "ZZ", "CZ", "CX", "XY", "CRZ", "CRY", "XX", "YY", "SWAP", "ISWAP"],
        help="Two-qubit gate used inside the compact QCNN block.",
    )
    parser.add_argument(
        "--scat-grid",
        type=int,
        default=None,
        help="Local scattering grid size. 4 means 4x4 regional tokens.",
    )
    parser.add_argument(
        "--compress-mode",
        default="flat",
        choices=["flat", "shared"],
        help="Use a flattened local compressor or a shared token-angle projector.",
    )
    parser.add_argument(
        "--frontend-mode",
        default="auto",
        choices=["auto", "global", "local", "conv", "shapelet"],
        help="Auto-pick the best low-parameter frontend per dataset, or force a specific frontend.",
    )
    parser.add_argument(
        "--conv-width",
        type=int,
        default=None,
        help="Token width used by the tiny convolutional tokenizer.",
    )
    parser.add_argument(
        "--shapelet-count",
        type=int,
        default=None,
        help="Number of learned spectral shapelets used by the shapelet frontend.",
    )
    parser.add_argument(
        "--shapelet-length",
        type=int,
        default=None,
        help="Requested spectral shapelet length along the band axis.",
    )
    parser.add_argument(
        "--shapelet-stats",
        default="compact4",
        choices=["compact4", "full6"],
        help="Statistic set for the shapelet frontend: compact4 or full6.",
    )
    parser.add_argument(
        "--summary-mode",
        default="auto",
        choices=["auto", "attn", "attn_mean", "attn_mean_max"],
        help="Classical skip summary used alongside the quantum output.",
    )
    parser.add_argument(
        "--angle-mode",
        default="auto",
        choices=["auto", "positive", "signed"],
        help="Map compressor outputs either to [0, pi] or a signed range.",
    )
    parser.add_argument(
        "--ablation",
        default="none",
        choices=[
            "none",
            "classical_only",
            "mlp_replace",
            "frozen_quantum",
            "no_iqp",
            "no_reupload",
            "fixed_z",
            "quantum_only",
            "mlp_only",
            "frozen_mlp",
            "frozen_rff",
        ],
        help=(
            "Ablation controls that isolate the quantum contribution. "
            "none keeps the full model. classical_only drops the quantum read-out. "
            "mlp_replace swaps the quantum branch for a parameter-matched MLP. "
            "frozen_quantum freezes the quantum weights after random init. "
            "no_iqp uses plain RY angle encoding instead of IQP. "
            "no_reupload encodes once instead of every round. "
            "fixed_z forces the Pauli-Z read-out. "
            "quantum_only feeds the classifier only the quantum read-out, with no "
            "angle or frontend skip, for a clean head-to-head. "
            "mlp_only is the parameter-matched classical counterpart of "
            "quantum_only, feeding the classifier only the matched MLP read-out."
        ),
    )
    parser.add_argument(
        "--no-fusion",
        action="store_true",
        help=(
            "Feed the classifier only the branch read-out (no angle or frontend "
            "skip). Combines with any branch ablation: none, frozen_quantum, "
            "frozen_rff, mlp_replace, frozen_mlp. --ablation quantum_only is "
            "the same as --ablation none --no-fusion."
        ),
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help=(
            "auto keeps the original behaviour (MPS if present, else CPU, with the "
            "circuit always simulated on CPU). cuda puts the whole model, circuit "
            "included, on the GPU; use it for the large-register scaling runs."
        ),
    )
    parser.add_argument(
        "--teacher",
        default="none",
        choices=["none", "far", "near"],
        help=(
            "Positive control. Replace the labels by those of a fixed teacher: the same "
            "compressor form and circuit at teacher parameters theta*, read out and mapped "
            "to K balanced classes by a fixed random linear map. far draws theta* "
            "independently of the run; near puts theta* at relative distance "
            "--teacher-radius from this seed's circuit initialization theta_0. Global "
            "frontend only; all arms of a seed see the same labels."
        ),
    )
    parser.add_argument("--teacher-radius", type=float, default=0.1,
                        help="near teacher: ||theta* - theta_0|| / ||theta_0||.")
    parser.add_argument("--teacher-seed", type=int, default=2026,
                        help="Seed of the teacher's compressor, read-out map, and direction.")
    parser.add_argument("--no-aug", action="store_true",
                        help="Train without augmentation (required with --teacher: the "
                             "labels are defined on the clean image).")
    parser.add_argument(
        "--run-tag",
        default="",
        help="Optional suffix for the output directory, useful during search.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used for splits, sampling, and training.",
    )
    return parser


ARGS = build_parser().parse_args()
QUICK = ARGS.quick
DS = ARGS.dataset
OVERHEAD_MODE = ARGS.overhead_mode
if OVERHEAD_MODE == "fair5":
    OVERHEAD_MODE = "seqnn5"
MEASURE_BASES = ARGS.measure_bases
GATE_TYPE = ARGS.gate_type.upper()
RUN_TAG = ARGS.run_tag.strip()
COMPRESS_MODE = ARGS.compress_mode
FRONTEND_MODE = ARGS.frontend_mode
EVAL_ONLY = ARGS.eval_only
MODEL_PATH = ARGS.model_path.strip()
TTA_MODE = ARGS.tta
if FRONTEND_MODE == "auto":
    FRONTEND_MODE = "shapelet"
if TTA_MODE == "auto":
    TTA_MODE = "rot4" if DS == "overhead" else "none"
if MEASURE_BASES == "auto":
    MEASURE_BASES = "xyz" if FRONTEND_MODE == "global" else "x"
if GATE_TYPE == "AUTO":
    if DS == "overhead":
        GATE_TYPE = "XY"
    elif DS == "sat6":
        GATE_TYPE = "YY"
    else:
        GATE_TYPE = "CRY"
SUMMARY_MODE = ARGS.summary_mode
if SUMMARY_MODE == "auto":
    SUMMARY_MODE = "attn"
ANGLE_MODE = ARGS.angle_mode
if ANGLE_MODE == "auto":
    ANGLE_MODE = "positive"
ABLATION = ARGS.ablation
if ABLATION == "fixed_z":
    MEASURE_BASES = "z"
# No fusion: the head sees the branch output alone. quantum_only and mlp_only
# are the original spellings of none+no-fusion and mlp_replace+no-fusion.
NO_FUSION = bool(ARGS.no_fusion) or ABLATION in ("quantum_only", "mlp_only")
if NO_FUSION and ABLATION == "classical_only":
    raise SystemExit("--no-fusion with --ablation classical_only leaves the head no input")
TEACHER = ARGS.teacher
if TEACHER != "none":
    if FRONTEND_MODE != "global":
        raise SystemExit("--teacher needs --frontend-mode global (a parameter-free descriptor)")
    if ABLATION not in ("none", "frozen_quantum", "frozen_rff", "classical_only", "mlp_replace", "frozen_mlp"):
        raise SystemExit("--teacher changes the circuit for no_iqp/no_reupload/fixed_z; use a branch control")
    if not ARGS.no_aug:
        raise SystemExit("--teacher requires --no-aug: teacher labels are defined on the clean image")


def get_device():
    if ARGS.device == "cuda":
        if not torch.cuda.is_available():
            raise SystemExit("--device cuda requested but no CUDA device is visible")
        print(f"Device: CUDA ({torch.cuda.get_device_name(0)}), circuit on GPU")
        return torch.device("cuda")
    if ARGS.device == "auto" and torch.backends.mps.is_available():
        print("Device: Apple Silicon (MPS)")
        return torch.device("mps")
    print("Device: CPU")
    return torch.device("cpu")


DEVICE = get_device()
# The circuit stays on CPU unless the whole run is on CUDA (original behaviour).
Q_DEVICE = DEVICE if DEVICE.type == "cuda" else torch.device("cpu")


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
BASE_DIR = Path(os.environ.get("S2Q_DATA_ROOT", REPO_ROOT / "data")).expanduser()
RESULTS_DIR = SCRIPT_DIR / "sq_seqnn_outputs"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

PATHS = {
    "overhead": {
        "train": BASE_DIR / "Overhead_MNIST" / "MNIST" / "version2" / "train",
        "test": BASE_DIR / "Overhead_MNIST" / "MNIST" / "version2" / "test",
    },
    "sat6": {
        "mat": BASE_DIR / "Sat6_dataset" / "sat-6-full.mat",
    },
    "so2sat": {
        "train": BASE_DIR / "So2Sat_LCZ42" / "training.h5",
        "val": BASE_DIR / "So2Sat_LCZ42" / "validation.h5",
        "test": BASE_DIR / "So2Sat_LCZ42" / "testing.h5",
        "train_geo": BASE_DIR / "So2Sat_LCZ42" / "training_geo.h5",
        "val_geo": BASE_DIR / "So2Sat_LCZ42" / "validation_geo.h5",
        "test_geo": BASE_DIR / "So2Sat_LCZ42" / "testing_geo.h5",
    },
}

OVERHEAD_FULL10 = [
    "car", "harbor", "helicopter", "oil_gas_field", "parking_lot",
    "plane", "runway_mark", "ship", "stadium", "storage_tank",
]
OVERHEAD_FAIR5 = [
    "car", "harbor", "parking_lot", "plane", "ship",
]

META = {
    "overhead": {
        "name": "Overhead MNIST",
        "classes": OVERHEAD_FULL10 if OVERHEAD_MODE == "full10" else OVERHEAD_FAIR5,
        "K": 10 if OVERHEAD_MODE == "full10" else 5,
        "C": 1,
        "size": 32,
        "seqnn_train_pool_total": 4443,
        "seqnn_val_total": 667,
        "seqnn_test_total": 557,
    },
    "sat6": {
        "name": "SAT-6",
        "classes": ["building", "barren_land", "trees", "grassland", "road", "water"],
        "K": 6,
        "C": 4,
        "size": 32,
        "n_train_per_class": 700,
        "n_val_per_class": 200,
        "n_test_per_class": 200,
    },
    "so2sat": {
        "name": "So2Sat LCZ42",
        "classes": [
            "urban_compact", "urban_open",
            "forest_trees", "low_vegetation", "bare_water",
        ],
        "K": 5,
        "C": 4,
        "size": 32,
        "n_train_per_class": 1200,
        "n_val_per_class": 400,
        "n_test_per_class": 400,
    },
}

SEN2_BANDS = [2, 1, 0, 6]  # B4, B3, B2, B8 -> RGBN
LCZ_TO_SEMANTIC = {
    0: 0, 1: 0, 2: 0,
    3: 1, 4: 1, 5: 1, 6: 1, 7: 1, 8: 1, 9: 1,
    10: 2, 11: 2, 12: 2,
    13: 3,
    14: 4, 15: 4, 16: 4,
}


CONFIG = {
    "SCAT_J": 2,
    "N_QUBITS": ARGS.n_qubits or 10,
    "N_REUP": ARGS.n_reupload if ARGS.n_reupload is not None else 2,
    "EPOCHS": ARGS.epochs if ARGS.epochs is not None else (8 if QUICK else 80),
    "BATCH_SIZE": ARGS.batch_size if ARGS.batch_size is not None else (24 if QUICK else 64),
    "LR": ARGS.lr if ARGS.lr is not None else 0.005,
    "LR_Q_FACTOR": ARGS.lr_q_factor if ARGS.lr_q_factor is not None else 0.1,
    "WEIGHT_DECAY": 1e-4,
    "PATIENCE": 6 if QUICK else 20,
    "VAL_SPLIT": 0.15,
    "MAX_TRAIN_SAMPLES": 500 if QUICK else None,
    "MAX_TEST_SAMPLES": 250 if QUICK else None,
    "LABEL_SMOOTH": ARGS.label_smooth if ARGS.label_smooth is not None else 0.05,
    "SEED": ARGS.seed,
    "CLS_HIDDEN": ARGS.cls_hidden or (10 if FRONTEND_MODE in {"global", "shapelet"} else 12),
    "SCAT_GRID": ARGS.scat_grid or 4,
    "CONV_WIDTH": ARGS.conv_width or 4,
    "SHAPELET_COUNT": ARGS.shapelet_count or 6,
    "SHAPELET_LENGTH": ARGS.shapelet_length or 4,
    "SHAPELET_STATS": ARGS.shapelet_stats,
}

N_QUBITS = CONFIG["N_QUBITS"]
N_REUP = CONFIG["N_REUP"]
QCNN_PAIR_PARAMS = 8
QCNN_PAIRS_PER_ROUND = N_QUBITS - 1
N_Q_PARAMS = QCNN_PAIRS_PER_ROUND * QCNN_PAIR_PARAMS * N_REUP
if MEASURE_BASES == "xyz":
    N_Q_OUT = 3 * N_QUBITS
elif MEASURE_BASES == "y":
    N_Q_OUT = N_QUBITS
elif MEASURE_BASES == "xz":
    N_Q_OUT = 2 * N_QUBITS
else:
    N_Q_OUT = N_QUBITS


def to_angles(logits):
    if ANGLE_MODE == "signed":
        return torch.tanh(logits) * (math.pi / 2.0)
    return torch.sigmoid(logits) * math.pi


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


set_seed(CONFIG["SEED"])


def iqr_norm(arr):
    out = arr.copy().astype(np.float32)
    for c in range(arr.shape[1]):
        ch = arr[:, c, :, :]
        p2, p98 = np.percentile(ch, 2), np.percentile(ch, 98)
        ch = np.clip(ch, p2, p98)
        denom = p98 - p2
        out[:, c, :, :] = (ch - p2) / denom if denom > 0 else np.zeros_like(ch)
    return out


def _sample_per_class(labels, counts, rng):
    selected = []
    for cls, n_needed in counts.items():
        cls_idx = np.where(labels == cls)[0]
        if len(cls_idx) < n_needed:
            raise ValueError(f"Class {cls} has only {len(cls_idx)} samples (<{n_needed}).")
        selected.append(rng.choice(cls_idx, size=n_needed, replace=False))
    idx = np.concatenate(selected)
    rng.shuffle(idx)
    return idx


def _decode_h5_strings(values):
    arr = np.asarray(values)
    if arr.dtype.kind in {"S", "O"}:
        return np.array([
            v.decode("utf-8") if isinstance(v, (bytes, np.bytes_)) else str(v)
            for v in arr
        ])
    return arr.astype(str)


def _limit_per_class(files, class_count, max_total):
    if max_total is None:
        return files
    return files[: max(1, max_total // max(class_count, 1))]


def _load_overhead_folder(base, classes, size, max_total=None):
    from PIL import Image

    ext = {".png", ".jpg", ".jpeg", ".bmp"}
    imgs, lbls = [], []
    for ci, cn in enumerate(classes):
        d = Path(base) / cn
        if not d.is_dir():
            print(f"  [WARN] Missing class folder: {d}")
            continue
        files = sorted([f for f in d.iterdir() if f.suffix.lower() in ext])
        files = _limit_per_class(files, len(classes), max_total)
        for f in files:
            try:
                img = Image.open(f).convert("L").resize(
                    (size, size), Image.BILINEAR
                )
                arr = np.array(img, dtype=np.float32)[np.newaxis, np.newaxis]
                imgs.append(arr)
                lbls.append(ci)
            except Exception:
                pass
        print(f"  {cn:>15}: {sum(1 for y in lbls if y == ci)}")
    return np.concatenate(imgs), np.array(lbls, dtype=np.int64)


def _cap_split_quick(x, y, max_total):
    if max_total is None or len(y) <= max_total:
        return x, y
    keep_idx, _ = train_test_split(
        np.arange(len(y)),
        train_size=max_total,
        stratify=y,
        random_state=CONFIG["SEED"],
    )
    return x[keep_idx], y[keep_idx]


def load_overhead(paths, meta):
    classes = meta["classes"]
    size = meta["size"]

    print("[TRAIN SOURCE]")
    x_train_all, y_train_all = _load_overhead_folder(
        paths["train"], classes, size,
        None if OVERHEAD_MODE == "seqnn5" else CONFIG["MAX_TRAIN_SAMPLES"],
    )
    print("[TEST SOURCE]")
    x_test_all, y_test_all = _load_overhead_folder(
        paths["test"], classes, size,
        None if OVERHEAD_MODE == "seqnn5" else CONFIG["MAX_TEST_SAMPLES"],
    )

    if OVERHEAD_MODE == "seqnn5":
        train_pool_total = meta["seqnn_train_pool_total"]
        val_total = meta["seqnn_val_total"]
        test_total = meta["seqnn_test_total"]

        train_pool_idx, _ = train_test_split(
            np.arange(len(y_train_all)),
            train_size=train_pool_total,
            stratify=y_train_all,
            random_state=CONFIG["SEED"],
        )
        test_pool_idx, _ = train_test_split(
            np.arange(len(y_test_all)),
            train_size=test_total,
            stratify=y_test_all,
            random_state=CONFIG["SEED"],
        )

        x_pool = x_train_all[train_pool_idx]
        y_pool = y_train_all[train_pool_idx]
        x_test = x_test_all[test_pool_idx]
        y_test = y_test_all[test_pool_idx]

        val_ratio = val_total / train_pool_total
        x_train, x_val, y_train, y_val = train_test_split(
            x_pool,
            y_pool,
            test_size=val_ratio,
            stratify=y_pool,
            random_state=CONFIG["SEED"],
        )

        if QUICK:
            x_train, y_train = _cap_split_quick(x_train, y_train, CONFIG["MAX_TRAIN_SAMPLES"])
            x_val, y_val = _cap_split_quick(x_val, y_val, max(1, CONFIG["MAX_TEST_SAMPLES"] // 3))
            x_test, y_test = _cap_split_quick(x_test, y_test, CONFIG["MAX_TEST_SAMPLES"])

        print(
            f"  SEQNN-style split"
            f"{' (quick-capped)' if QUICK else ''}: "
            f"train={len(x_train)} val={len(x_val)} test={len(x_test)}"
        )
        return iqr_norm(x_train), y_train, iqr_norm(x_val), y_val, iqr_norm(x_test), y_test

    return iqr_norm(x_train_all), y_train_all, None, None, iqr_norm(x_test_all), y_test_all


def load_sat6(paths, meta):
    import scipy.io as sio

    mat_path = paths["mat"]
    if not mat_path.exists():
        raise FileNotFoundError(f"SAT-6 file not found: {mat_path}")

    print(f"  Loading {mat_path} ...")
    data = sio.loadmat(str(mat_path))
    rng = np.random.RandomState(CONFIG["SEED"])

    tr_x = data["train_x"].transpose(3, 2, 0, 1).astype(np.float32) / 255.0
    tr_y = np.argmax(data["train_y"].T, axis=1).astype(np.int64)
    te_x = data["test_x"].transpose(3, 2, 0, 1).astype(np.float32) / 255.0
    te_y = np.argmax(data["test_y"].T, axis=1).astype(np.int64)

    pad = (meta["size"] - 28) // 2
    tr_x = np.pad(tr_x, ((0, 0), (0, 0), (pad, pad), (pad, pad)), mode="constant")
    te_x = np.pad(te_x, ((0, 0), (0, 0), (pad, pad), (pad, pad)), mode="constant")

    train_counts = {
        cls: meta["n_train_per_class"] + meta["n_val_per_class"]
        for cls in range(meta["K"])
    }
    test_counts = {
        cls: meta["n_test_per_class"]
        for cls in range(meta["K"])
    }
    train_pool_idx = _sample_per_class(tr_y, train_counts, rng)
    test_idx = _sample_per_class(te_y, test_counts, rng)

    pooled_x = tr_x[train_pool_idx]
    pooled_y = tr_y[train_pool_idx]
    val_ratio = meta["n_val_per_class"] / (meta["n_train_per_class"] + meta["n_val_per_class"])
    tr_x, val_x, tr_y, val_y = train_test_split(
        pooled_x,
        pooled_y,
        test_size=val_ratio,
        stratify=pooled_y,
        random_state=CONFIG["SEED"],
    )
    te_x = te_x[test_idx]
    te_y = te_y[test_idx]

    if QUICK:
        tr_x, tr_y = _cap_split_quick(tr_x, tr_y, CONFIG["MAX_TRAIN_SAMPLES"])
        val_x, val_y = _cap_split_quick(val_x, val_y, max(1, CONFIG["MAX_TEST_SAMPLES"] // 3))
        te_x, te_y = _cap_split_quick(te_x, te_y, CONFIG["MAX_TEST_SAMPLES"])

    print(f"  SAT-6 protocol{' (quick-capped)' if QUICK else ''}: train={len(tr_x)} val={len(val_x)} test={len(te_x)}")
    return iqr_norm(tr_x), tr_y, iqr_norm(val_x), val_y, iqr_norm(te_x), te_y


def load_so2sat(paths, meta):
    import h5py

    def read_city_split(h5_path, geo_path, target_cities):
        if not h5_path.exists():
            raise FileNotFoundError(f"So2Sat file not found: {h5_path}")
        if not geo_path.exists():
            raise FileNotFoundError(f"So2Sat geo file not found: {geo_path}")
        with h5py.File(str(geo_path), "r") as gf:
            cities = _decode_h5_strings(gf["city"][:])
        city_mask = np.isin(cities, target_cities)
        with h5py.File(str(h5_path), "r") as f:
            labels_oh = np.asarray(f["label"][city_mask], dtype=np.float32)
            sen2 = np.asarray(f["sen2"][city_mask], dtype=np.float32)

        lcz_raw = np.argmax(labels_oh, axis=1)
        mapped = np.array([LCZ_TO_SEMANTIC.get(int(label), -1) for label in lcz_raw], dtype=np.int64)
        valid = mapped >= 0
        mapped = mapped[valid]
        sen2 = sen2[valid]

        imgs = sen2[:, :, :, SEN2_BANDS]
        imgs = imgs.transpose(0, 3, 1, 2).astype(np.float32)
        for band_idx in range(imgs.shape[1]):
            band = imgs[:, band_idx]
            bmin = float(band.min())
            bmax = float(band.max())
            imgs[:, band_idx] = (band - bmin) / (bmax - bmin + 1e-8)
        return imgs.astype(np.float32), mapped

    rng = np.random.RandomState(CONFIG["SEED"])
    print("[TRAIN/CITY SPLITS]")
    imgs_bc, lbls_bc = read_city_split(paths["train"], paths["train_geo"], ["berlin", "cologne"])
    imgs_mv, lbls_mv = read_city_split(paths["val"], paths["val_geo"], ["munich"])
    imgs_mt, lbls_mt = read_city_split(paths["test"], paths["test_geo"], ["munich"])

    all_imgs = np.concatenate([imgs_bc, imgs_mv, imgs_mt], axis=0)
    all_lbls = np.concatenate([lbls_bc, lbls_mv, lbls_mt], axis=0)
    total_per_class = meta["n_train_per_class"] + meta["n_val_per_class"] + meta["n_test_per_class"]
    sampled_idx = _sample_per_class(
        all_lbls,
        {cls: total_per_class for cls in range(meta["K"])},
        rng,
    )
    imgs_all = all_imgs[sampled_idx]
    lbls_all = all_lbls[sampled_idx]

    train_ratio = meta["n_train_per_class"] / total_per_class
    tr_x, rest_x, tr_y, rest_y = train_test_split(
        imgs_all,
        lbls_all,
        test_size=1.0 - train_ratio,
        stratify=lbls_all,
        random_state=CONFIG["SEED"],
    )
    val_ratio = meta["n_val_per_class"] / (meta["n_val_per_class"] + meta["n_test_per_class"])
    val_x, te_x, val_y, te_y = train_test_split(
        rest_x,
        rest_y,
        test_size=1.0 - val_ratio,
        stratify=rest_y,
        random_state=CONFIG["SEED"],
    )

    if QUICK:
        tr_x, tr_y = _cap_split_quick(tr_x, tr_y, CONFIG["MAX_TRAIN_SAMPLES"])
        val_x, val_y = _cap_split_quick(val_x, val_y, max(1, CONFIG["MAX_TEST_SAMPLES"] // 3))
        te_x, te_y = _cap_split_quick(te_x, te_y, CONFIG["MAX_TEST_SAMPLES"])

    print(f"  So2Sat protocol{' (quick-capped)' if QUICK else ''}: train={len(tr_x)} val={len(val_x)} test={len(te_x)}  Channels: {tr_x.shape[1]}")
    return iqr_norm(tr_x), tr_y, iqr_norm(val_x), val_y, iqr_norm(te_x), te_y


def load_data(ds):
    meta = META[ds]
    paths = PATHS[ds]
    print("=" * 60)
    print(f"Dataset: {meta['name']}  |  quick={QUICK}")
    if ds == "overhead":
        print(f"Overhead mode: {OVERHEAD_MODE}")
    print("=" * 60)

    for key, value in paths.items():
        print(f"  {key}: {value}  [{'✓' if Path(value).exists() else '✗'}]")

    if ds == "overhead":
        tr_x, tr_y, val_x, val_y, te_x, te_y = load_overhead(paths, meta)
        if OVERHEAD_MODE == "seqnn5":
            print(f"\n  Train: {len(tr_x)}  Val: {len(val_x)}  Test: {len(te_x)}")
            return tr_x, tr_y, val_x, val_y, te_x, te_y
    elif ds == "sat6":
        tr_x, tr_y, val_x, val_y, te_x, te_y = load_sat6(paths, meta)
        print(f"\n  Train: {len(tr_x)}  Val: {len(val_x)}  Test: {len(te_x)}")
        return tr_x, tr_y, val_x, val_y, te_x, te_y
    else:
        tr_x, tr_y, val_x, val_y, te_x, te_y = load_so2sat(paths, meta)
        print(f"\n  Train: {len(tr_x)}  Val: {len(val_x)}  Test: {len(te_x)}")
        return tr_x, tr_y, val_x, val_y, te_x, te_y

    tr_x, val_x, tr_y, val_y = train_test_split(
        tr_x,
        tr_y,
        test_size=CONFIG["VAL_SPLIT"],
        stratify=tr_y,
        random_state=CONFIG["SEED"],
    )

    print(f"\n  Train: {len(tr_x)}  Val: {len(val_x)}  Test: {len(te_x)}")
    return tr_x, tr_y, val_x, val_y, te_x, te_y


def augment(img):
    from scipy.ndimage import rotate as sp_rot

    img = img.copy()
    if np.random.rand() > 0.5:
        img = np.flip(img, 2).copy()
    if np.random.rand() > 0.5:
        img = np.flip(img, 1).copy()
    angle = np.random.uniform(-20, 20)
    img = np.stack([
        sp_rot(img[c], angle, reshape=False, mode="reflect")
        for c in range(img.shape[0])
    ])
    if np.random.rand() > 0.5:
        img += np.random.normal(0, 0.02, img.shape).astype(np.float32)
    for c in range(img.shape[0]):
        img[c] *= np.random.uniform(0.85, 1.15)
    return np.clip(img, 0, 1).astype(np.float32)


class RSDataset(Dataset):
    def __init__(self, X, y, aug=False):
        self.X = X
        self.y = torch.from_numpy(y).long()
        self.aug = aug

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x = augment(self.X[idx]) if self.aug and np.random.rand() > 0.4 else self.X[idx]
        return torch.from_numpy(x.copy()).float(), self.y[idx]


def balanced_loader(X, y, batch_size, aug=False):
    counts = np.bincount(y, minlength=y.max() + 1)
    weights = (1.0 / np.maximum(counts, 1).astype(np.float32))[y]
    sampler = WeightedRandomSampler(torch.from_numpy(weights), len(y), replacement=True)
    return DataLoader(RSDataset(X, y, aug), batch_size=batch_size, sampler=sampler, num_workers=0)


def plain_loader(X, y, batch_size):
    return DataLoader(RSDataset(X, y, False), batch_size=batch_size, shuffle=False, num_workers=0)


class GlobalScatteringTransform(nn.Module):
    """Compact global scattering-style summary used by the best saved full run."""

    def __init__(self, C, J=2, size=32):
        super().__init__()
        self.C = C
        self.J = J
        self.size = size
        self.out_dim = C * (1 + 2 * J)
        self._build_filters()

    def _build_filters(self):
        filters = []
        for j in range(self.J):
            sigma = 2 ** (j + 1)
            k = int(4 * sigma + 1) | 1
            pad = k // 2
            ax = torch.arange(-pad, pad + 1, dtype=torch.float32)
            g = torch.exp(-(ax ** 2) / (2 * sigma ** 2))
            g = g / g.sum()
            kern = g.outer(g).unsqueeze(0).unsqueeze(0)
            filters.append((kern, pad))
        for j, (kern, pad) in enumerate(filters):
            self.register_buffer(f"filter_{j}", kern)
            self.register_buffer(f"pad_{j}", torch.tensor(pad, dtype=torch.long))

    def forward(self, x):
        feats = []
        for c in range(x.shape[1]):
            xc = x[:, c : c + 1]
            feats.append(xc.mean(dim=[2, 3]))
            prev = xc
            for j in range(self.J):
                kern = getattr(self, f"filter_{j}")
                pad = int(getattr(self, f"pad_{j}").item())
                blurred = F.conv2d(prev, kern, padding=pad)
                edge_map = (prev - blurred).abs()
                feats.append(edge_map.mean(dim=[2, 3]))
                texture = F.conv2d(edge_map, kern, padding=pad)
                feats.append(texture.mean(dim=[2, 3]))
                prev = blurred
        return torch.cat(feats, dim=1)


class ScatteringTransform(nn.Module):
    """
    Fixed local multi-scale scattering-style tokenization.

    Returns tokens of shape (B, grid^2, C*(1+2J)).
    """

    def __init__(self, C, J=2, size=32, grid=4):
        super().__init__()
        self.C = C
        self.J = J
        self.size = size
        self.grid = grid
        self.token_dim = C * (1 + 2 * J)
        self.num_tokens = grid * grid
        self.out_dim = self.token_dim * self.num_tokens
        self._build_filters()

    def _build_filters(self):
        filters = []
        for j in range(self.J):
            sigma = 2 ** (j + 1)
            k = int(4 * sigma + 1) | 1
            pad = k // 2
            ax = torch.arange(-pad, pad + 1, dtype=torch.float32)
            g = torch.exp(-(ax ** 2) / (2 * sigma ** 2))
            g = g / g.sum()
            kern = g.outer(g).unsqueeze(0).unsqueeze(0)
            filters.append((kern, pad))
        for j, (kern, pad) in enumerate(filters):
            self.register_buffer(f"filter_{j}", kern)
            self.register_buffer(f"pad_{j}", torch.tensor(pad, dtype=torch.long))

    def forward(self, x):
        token_groups = []
        for c in range(x.shape[1]):
            xc = x[:, c : c + 1]
            maps = [xc]
            prev = xc
            for j in range(self.J):
                kern = getattr(self, f"filter_{j}")
                pad = int(getattr(self, f"pad_{j}").item())
                blurred = F.conv2d(prev, kern, padding=pad)
                edge_map = (prev - blurred).abs()
                texture = F.conv2d(edge_map, kern, padding=pad)
                maps.extend([edge_map, texture])
                prev = blurred

            channel_tokens = []
            for m in maps:
                pooled = F.adaptive_avg_pool2d(m, (self.grid, self.grid))
                channel_tokens.append(pooled.flatten(2).transpose(1, 2))
            token_groups.append(torch.cat(channel_tokens, dim=-1))

        return torch.cat(token_groups, dim=-1)


class TinyConvTokenizer(nn.Module):
    """
    Tiny learned tokenizer with depthwise-separable convolutions.

    Returns tokens of shape (B, grid^2, width).
    """

    def __init__(self, C, size=32, grid=4, width=6):
        super().__init__()
        self.C = C
        self.size = size
        self.grid = grid
        self.width = width
        self.token_dim = width
        self.num_tokens = grid * grid
        self.out_dim = self.token_dim * self.num_tokens
        self.dw1 = nn.Conv2d(C, C, kernel_size=3, padding=1, groups=C, bias=False)
        self.pw1 = nn.Conv2d(C, width, kernel_size=1, bias=True)
        self.dw2 = nn.Conv2d(width, width, kernel_size=3, padding=1, groups=width, bias=False)
        self.pw2 = nn.Conv2d(width, width, kernel_size=1, bias=True)
        nn.init.kaiming_normal_(self.dw1.weight, nonlinearity="relu")
        nn.init.kaiming_normal_(self.pw1.weight, nonlinearity="relu")
        nn.init.zeros_(self.pw1.bias)
        nn.init.kaiming_normal_(self.dw2.weight, nonlinearity="relu")
        nn.init.kaiming_normal_(self.pw2.weight, nonlinearity="relu")
        nn.init.zeros_(self.pw2.bias)

    def forward(self, x):
        x = F.gelu(self.pw1(self.dw1(x)))
        x = F.gelu(self.pw2(self.dw2(x)))
        x = F.adaptive_avg_pool2d(x, (self.grid, self.grid))
        return x.flatten(2).transpose(1, 2)


class SpectralShapeletStats(nn.Module):
    """
    Learn short spectral motifs along the band axis, then summarize the
    resulting response maps with parameter-free spatial statistics.

    This is most meaningful when the channel axis is an ordered spectrum
    (e.g. multispectral Sentinel-2 bands), but it still runs on the current
    compact protocols by shrinking the effective shapelet length to the number
    of available bands.
    """

    def __init__(self, C, count=6, length=4, stat_mode="compact4"):
        super().__init__()
        self.C = C
        self.count = count
        self.length = max(1, min(length, C))
        self.stat_mode = stat_mode
        self.n_stats = 4 if stat_mode == "compact4" else 6
        self.out_dim = count * self.n_stats
        self.shapelets = nn.Parameter(torch.empty(count, self.length))
        self.log_tau = nn.Parameter(torch.zeros(1))
        nn.init.uniform_(self.shapelets, 0.0, 1.0)

    @staticmethod
    def _corr_2d(lhs, rhs):
        lhs_f = lhs.flatten(2)
        rhs_f = rhs.flatten(2)
        lhs_f = lhs_f - lhs_f.mean(dim=2, keepdim=True)
        rhs_f = rhs_f - rhs_f.mean(dim=2, keepdim=True)
        num = (lhs_f * rhs_f).mean(dim=2)
        den = torch.sqrt(
            lhs_f.square().mean(dim=2).clamp_min(1e-6)
            * rhs_f.square().mean(dim=2).clamp_min(1e-6)
        )
        return num / den.clamp_min(1e-6)

    def _response_maps(self, x):
        # x: (B, C, H, W) -> windows: (B, n_pos, H, W, L)
        windows = x.unfold(dimension=1, size=self.length, step=1)
        diffs = windows.unsqueeze(1) - self.shapelets.view(1, self.count, 1, 1, 1, self.length)
        dists = diffs.square().sum(dim=-1)
        best = dists.min(dim=2).values
        tau = F.softplus(self.log_tau).clamp_min(1e-4)
        return torch.exp(-best / (tau * tau))

    def forward(self, x):
        maps = self._response_maps(x)
        flat = maps.flatten(2)
        stats = [
            maps.mean(dim=(2, 3)),
            flat.std(dim=2, unbiased=False),
            self._corr_2d(maps[:, :, :, :-1], maps[:, :, :, 1:]),
            self._corr_2d(maps[:, :, :-1, :], maps[:, :, 1:, :]),
        ]
        if self.stat_mode == "full6":
            p90_idx = min(
                flat.shape[2] - 1,
                max(0, int(math.ceil(0.9 * flat.shape[2])) - 1),
            )
            p90 = flat.sort(dim=2).values[:, :, p90_idx]
            stats.append(maps.amax(dim=(2, 3)))
            stats.append(p90)
        return torch.cat(stats, dim=1)


class RegionAttentionPool(nn.Module):
    """Tiny attention pool over local scattering tokens."""

    def __init__(self, token_dim):
        super().__init__()
        self.score = nn.Linear(token_dim, 1)
        nn.init.xavier_uniform_(self.score.weight)
        nn.init.zeros_(self.score.bias)

    def forward(self, tokens):
        logits = self.score(tokens).squeeze(-1)
        weights = torch.softmax(logits, dim=1)
        pooled = (tokens * weights.unsqueeze(-1)).sum(dim=1)
        return pooled, weights


class AngleCompressor(nn.Module):
    def __init__(self, in_dim, n_qubits=N_QUBITS):
        super().__init__()
        self.fc = nn.Linear(in_dim, n_qubits)
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def forward(self, s):
        s = F.layer_norm(s, (s.shape[-1],))
        return self.fc(s)


class MLPQuantumReplacement(nn.Module):
    """
    Classical control for the mlp_replace ablation.

    Replaces the quantum branch with a small MLP that maps the angle vector to a
    read-out of the same width as the quantum output, sized so that its trainable
    parameter count matches the quantum branch as closely as possible. This keeps
    the classifier input shape and the overall budget comparable to the full
    model, so any accuracy gap reflects the quantum circuit rather than capacity.
    """

    def __init__(self, in_dim, out_dim, target_params=N_Q_PARAMS):
        super().__init__()
        hidden = max(1, round((target_params - out_dim) / (in_dim + 1 + out_dim)))
        self.hidden = hidden
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, out_dim),
        )
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, alpha):
        return self.net(alpha)


class RandomFourierFeatures(nn.Module):
    """
    Classical control for the frozen_rff ablation.

    Maps the angle vector to cos(W alpha + b) with W ~ N(0, 1) and
    b ~ U(0, 2 pi) drawn once from the seeded generator and stored as buffers,
    so the branch has zero trainable parameters and an output in [-1, 1] like
    a vector of Pauli expectations. If this matches the trained circuit, the
    head only needs some fixed nonlinear projection of alpha, not a quantum
    one and not a trained one.
    """

    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.register_buffer("W", torch.randn(out_dim, in_dim))
        self.register_buffer("b", torch.rand(out_dim) * (2.0 * math.pi))

    def forward(self, alpha):
        return torch.cos(alpha @ self.W.t() + self.b)


dev = qml.device("default.qubit", wires=N_QUBITS)
print(f"PennyLane {qml.__version__}: default.qubit {N_QUBITS} qubits, {N_REUP} re-upload rounds")


def iqp_encode(f, include_hadamards=True):
    if include_hadamards:
        for q in range(N_QUBITS):
            qml.Hadamard(wires=q)
    for q in range(N_QUBITS):
        qml.RZ(2.0 * f[q], wires=q)
    for q in range(N_QUBITS - 1):
        qml.CNOT(wires=[q, q + 1])
        qml.RZ(2.0 * f[q] * f[q + 1], wires=q + 1)
        qml.CNOT(wires=[q, q + 1])


def _apply_entangler(gate_type, param, q0, q1):
    if gate_type == "ZZ":
        qml.IsingZZ(param, wires=[q0, q1])
    elif gate_type == "XY":
        qml.IsingXY(param, wires=[q0, q1])
    elif gate_type == "XX":
        qml.IsingXX(param, wires=[q0, q1])
    elif gate_type == "YY":
        qml.IsingYY(param, wires=[q0, q1])
    elif gate_type == "CRZ":
        qml.CRZ(param, wires=[q0, q1])
    elif gate_type == "CRY":
        qml.CRY(param, wires=[q0, q1])
    elif gate_type == "CZ":
        qml.CZ(wires=[q0, q1])
    elif gate_type == "CX":
        qml.CNOT(wires=[q0, q1])
    elif gate_type == "SWAP":
        qml.SWAP(wires=[q0, q1])
    elif gate_type == "ISWAP":
        qml.ISWAP(wires=[q0, q1])
    else:
        raise ValueError(f"Unsupported gate type: {gate_type}")


def qcnn_pair(w, q0, q1):
    # A compact shared QCNN kernel with configurable entanglers.
    qml.RY(w[0], wires=q0)
    qml.RY(w[1], wires=q1)
    _apply_entangler(GATE_TYPE, w[2], q0, q1)
    qml.RX(w[3], wires=q0)
    qml.RX(w[4], wires=q1)
    _apply_entangler(GATE_TYPE, w[5], q0, q1)
    qml.RY(w[6], wires=q0)
    qml.RY(w[7], wires=q1)


def qcnn_sweep(weights, ptr, start_idx):
    # Even-odd scheduling improves information mixing across the chain.
    for q in range(start_idx, N_QUBITS - 1, 2):
        qcnn_pair(weights[ptr : ptr + QCNN_PAIR_PARAMS], q, q + 1)
        ptr += QCNN_PAIR_PARAMS
    return ptr


def _encode(features, first_round):
    # no_iqp swaps the IQP map for a plain RY angle embedding to test whether
    # the IQP structure itself matters, while keeping the data path intact.
    if ABLATION == "no_iqp":
        for q in range(N_QUBITS):
            qml.RY(2.0 * features[q], wires=q)
    else:
        iqp_encode(features, include_hadamards=first_round)


@qml.qnode(dev, diff_method="backprop", interface="torch")
def _circuit(features, weights):
    ptr = 0
    for upload_idx in range(N_REUP):
        # no_reupload encodes the data only once and then applies every sweep.
        if not (ABLATION == "no_reupload" and upload_idx > 0):
            _encode(features, first_round=(upload_idx == 0))
        ptr = qcnn_sweep(weights, ptr, 0)
        ptr = qcnn_sweep(weights, ptr, 1)
    if MEASURE_BASES == "x":
        return [qml.expval(qml.PauliX(i)) for i in range(N_QUBITS)]
    if MEASURE_BASES == "y":
        return [qml.expval(qml.PauliY(i)) for i in range(N_QUBITS)]
    if MEASURE_BASES == "z":
        return [qml.expval(qml.PauliZ(i)) for i in range(N_QUBITS)]
    if MEASURE_BASES == "xz":
        x = [qml.expval(qml.PauliX(i)) for i in range(N_QUBITS)]
        z = [qml.expval(qml.PauliZ(i)) for i in range(N_QUBITS)]
        return x + z
    z = [qml.expval(qml.PauliZ(i)) for i in range(N_QUBITS)]
    x = [qml.expval(qml.PauliX(i)) for i in range(N_QUBITS)]
    y = [qml.expval(qml.PauliY(i)) for i in range(N_QUBITS)]
    return z + x + y


try:
    _batch = torch.vmap(_circuit, in_dims=(0, None), randomness="same")
    _VMAP = True
    print("torch.vmap: ON")
except Exception:
    _VMAP = False
    print("torch.vmap: OFF")


def q_forward(alpha_batch, weights):
    if _VMAP:
        out = _batch(alpha_batch, weights)
        if isinstance(out, (list, tuple)):
            return torch.stack(out, dim=1)
        return out
    return torch.stack([
        torch.stack(_circuit(alpha_batch[b], weights))
        for b in range(alpha_batch.shape[0])
    ])


class SQ_SEQNN(nn.Module):
    def __init__(self, C, size, K, J=CONFIG["SCAT_J"]):
        super().__init__()
        self.frontend_mode = FRONTEND_MODE
        self.summary_mode = SUMMARY_MODE
        # classical_only removes the quantum read-out from the fused vector.
        q_out_dim = 0 if ABLATION == "classical_only" else N_Q_OUT
        if self.frontend_mode == "global":
            self.scat = GlobalScatteringTransform(C, J, size)
            scat_dim = self.scat.out_dim
            self.region_pool = None
            self.compress = AngleCompressor(scat_dim, N_QUBITS)
            in_cls = q_out_dim + N_QUBITS + scat_dim
        elif self.frontend_mode == "shapelet":
            self.scat = SpectralShapeletStats(
                C,
                count=CONFIG["SHAPELET_COUNT"],
                length=CONFIG["SHAPELET_LENGTH"],
                stat_mode=CONFIG["SHAPELET_STATS"],
            )
            shapelet_dim = self.scat.out_dim
            self.region_pool = None
            self.compress = AngleCompressor(shapelet_dim, N_QUBITS)
            in_cls = q_out_dim + N_QUBITS + shapelet_dim
        else:
            if self.frontend_mode == "conv":
                self.scat = TinyConvTokenizer(
                    C,
                    size=size,
                    grid=CONFIG["SCAT_GRID"],
                    width=CONFIG["CONV_WIDTH"],
                )
            else:
                self.scat = ScatteringTransform(C, J, size, grid=CONFIG["SCAT_GRID"])
            token_dim = self.scat.token_dim
            flat_scat_dim = self.scat.out_dim
            self.region_pool = RegionAttentionPool(token_dim)
            if COMPRESS_MODE == "flat":
                self.compress = AngleCompressor(flat_scat_dim, N_QUBITS)
            else:
                self.compress = AngleCompressor(token_dim, N_QUBITS)
            local_summary_dim = token_dim
            if self.summary_mode == "attn_mean":
                local_summary_dim = 2 * token_dim
            elif self.summary_mode == "attn_mean_max":
                local_summary_dim = 3 * token_dim
            in_cls = q_out_dim + N_QUBITS + local_summary_dim
        # The isolation controls feed the classifier only the processed stream,
        # dropping the angle vector and the frontend skip descriptor, so that
        # quantum_only and mlp_only differ in nothing but the branch that maps
        # the angle vector to the read-out. This is the clean head-to-head test
        # of the quantum representation against a parameter-matched classical one.
        if NO_FUSION:
            in_cls = N_Q_OUT
        # The quantum weights are absent when there is no quantum branch.
        if ABLATION in ("classical_only", "mlp_replace", "mlp_only", "frozen_mlp", "frozen_rff"):
            self.register_parameter("q_w", None)
        else:
            self.q_w = nn.Parameter(torch.empty(N_Q_PARAMS).uniform_(0, 2 * math.pi))
            if ABLATION == "frozen_quantum":
                self.q_w.requires_grad_(False)
        if ABLATION in ("mlp_replace", "mlp_only", "frozen_mlp"):
            self.q_mlp = MLPQuantumReplacement(N_QUBITS, N_Q_OUT)
            if ABLATION == "frozen_mlp":
                for p_ in self.q_mlp.parameters():
                    p_.requires_grad_(False)
        elif ABLATION == "frozen_rff":
            self.q_mlp = RandomFourierFeatures(N_QUBITS, N_Q_OUT)
        else:
            self.q_mlp = None
        # Keep the classifier initialization identical across ablations for a
        # given seed. The classical_only and mlp_replace branches consume a
        # different amount of the random stream when building the quantum branch
        # above, which would otherwise hand the classifier a different random
        # init and confound the paired comparison. Re-seeding here makes the head
        # depend only on the seed, so a paired difference reflects the quantum
        # branch alone.
        torch.manual_seed(CONFIG["SEED"] + 9973)
        hidden = CONFIG["CLS_HIDDEN"]
        self.clf = nn.Sequential(
            nn.LayerNorm(in_cls),
            nn.Linear(in_cls, hidden),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, K),
        )
        nn.init.xavier_uniform_(self.clf[1].weight)
        nn.init.zeros_(self.clf[1].bias)
        nn.init.xavier_uniform_(self.clf[4].weight)
        nn.init.zeros_(self.clf[4].bias)

    def _quantum(self, alpha, x):
        if ABLATION in ("mlp_replace", "mlp_only", "frozen_mlp", "frozen_rff"):
            return self.q_mlp(alpha)
        a_q = alpha.to(Q_DEVICE, dtype=torch.float32)
        w_q = self.q_w.to(Q_DEVICE, dtype=torch.float32)
        return q_forward(a_q, w_q).float().to(x.device)

    def forward(self, x):
        if self.frontend_mode in ("global", "shapelet"):
            s = self.scat(x)
            alpha = to_angles(self.compress(s))
            if NO_FUSION:
                h = self._quantum(alpha, x)
            elif ABLATION == "classical_only":
                h = torch.cat([alpha, s], dim=1)
            else:
                q = self._quantum(alpha, x)
                h = torch.cat([q, alpha, s], dim=1)
        else:
            tokens = self.scat(x)
            pooled, weights = self.region_pool(tokens)
            if COMPRESS_MODE == "flat":
                alpha = to_angles(self.compress(tokens.flatten(1)))
            else:
                alpha_tokens = self.compress(tokens)
                alpha = to_angles((alpha_tokens * weights.unsqueeze(-1)).sum(dim=1))
            summary_parts = [pooled]
            if self.summary_mode in {"attn_mean", "attn_mean_max"}:
                summary_parts.append(tokens.mean(dim=1))
            if self.summary_mode == "attn_mean_max":
                summary_parts.append(tokens.amax(dim=1))
            local_summary = torch.cat(summary_parts, dim=1)
            if NO_FUSION:
                h = self._quantum(alpha, x)
            elif ABLATION == "classical_only":
                h = torch.cat([alpha, local_summary], dim=1)
            else:
                q = self._quantum(alpha, x)
                h = torch.cat([q, alpha, local_summary], dim=1)
        return self.clf(h)

    def count_params(self):
        scat = sum(p.numel() for p in self.scat.parameters())
        pool = 0 if self.region_pool is None else sum(p.numel() for p in self.region_pool.parameters())
        comp = sum(p.numel() for p in self.compress.parameters())
        q = 0 if self.q_w is None else self.q_w.numel()
        qmlp = 0 if self.q_mlp is None else sum(p.numel() for p in self.q_mlp.parameters())
        clf = sum(p.numel() for p in self.clf.parameters())
        return {
            "scattering": scat,
            "region_pool": pool,
            "compressor": comp,
            "quantum": q,
            "quantum_mlp": qmlp,
            "classifier": clf,
            "total": scat + pool + comp + q + qmlp + clf,
        }


class EarlyStopping:
    def __init__(self, patience):
        self.patience = patience
        self.best = -1.0
        self.count = 0
        self.state = None

    def __call__(self, acc, model):
        if acc > self.best + 1e-3:
            self.best = acc
            self.count = 0
            self.state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            self.count += 1
        return self.count >= self.patience

    def restore(self, model):
        if self.state is not None:
            model.load_state_dict(self.state)


def replicate_theta0(rng_state, scat_dim):
    """The circuit initialization theta_0 that SQ_SEQNN draws for this seed, computed on a
    forked RNG from the state just before the model is built: AngleCompressor first (Linear
    init plus xavier), then q_w ~ U[0, 2 pi). The global stream is left untouched, so every
    arm is initialized exactly as in the main campaign."""
    with torch.random.fork_rng(devices=[]):
        torch.set_rng_state(rng_state)
        AngleCompressor(scat_dim, N_QUBITS)
        return torch.empty(N_Q_PARAMS).uniform_(0, 2 * math.pi)


def teacher_relabel(splits, K, theta0):
    """Positive control: labels of a fixed teacher that has the student's own form.

    s = global scattering descriptor of the clean image (parameter free), alpha* =
    pi * sigmoid(W* LayerNorm(s) + b*), q* = circuit(alpha*; theta*) with the run's gate
    and read-out, label = argmax(V* q* + c). W*, V*, and the direction of theta* come from
    --teacher-seed; c equalizes the class counts over all images. The student can represent
    the teacher exactly, so a trained circuit is able to fit these labels; whether the frozen
    circuit, random features, or no circuit can is what the control measures."""
    g = torch.Generator().manual_seed(ARGS.teacher_seed)
    C = splits[0].shape[1]
    scat = GlobalScatteringTransform(C, CONFIG["SCAT_J"])
    d = scat.out_dim
    bound = math.sqrt(6.0 / (d + N_QUBITS))                         # xavier_uniform, as the student
    W = (torch.rand(N_QUBITS, d, generator=g) * 2 - 1) * bound
    V = torch.randn(K, N_Q_OUT, generator=g)
    direction = torch.randn(N_Q_PARAMS, generator=g)
    far_theta = torch.rand(N_Q_PARAMS, generator=g) * (2 * math.pi)
    if TEACHER == "far":
        theta = far_theta
    else:
        theta = theta0 + ARGS.teacher_radius * theta0.norm() * direction / direction.norm()
    feats, angles, qs = [], [], []
    with torch.no_grad():
        for X in splits:
            S, A, Q = [], [], []
            for i in range(0, len(X), 256):
                x = torch.from_numpy(np.ascontiguousarray(X[i:i + 256])).float()
                s = scat(x)
                a = torch.sigmoid(F.layer_norm(s, (d,)) @ W.t()) * math.pi
                q = q_forward(a.to(Q_DEVICE), theta.to(Q_DEVICE)).float().cpu()
                S.append(s); A.append(a); Q.append(q)
            feats.append(torch.cat(S)); angles.append(torch.cat(A)); qs.append(torch.cat(Q))
    logits = [q @ V.t() for q in qs]
    L = torch.cat(logits)
    c = torch.zeros(K)
    target = len(L) / K
    step = 0.5 * float(L.std())
    for _ in range(2000):                                          # equalize class counts
        cnt = torch.bincount((L + c).argmax(1), minlength=K).float()
        if (cnt - target).abs().max() <= max(1.0, 0.01 * target):
            break
        c -= step * torch.log((cnt + 1.0) / target)
        step *= 0.995
    ys = [(lg + c).argmax(1).numpy().astype(np.int64) for lg in logits]
    counts = np.bincount(np.concatenate(ys), minlength=K)
    assert counts.min() >= 0.8 * target, f"teacher classes unbalanced: {counts}"

    # How hard is the task? Linear read-outs, train -> test, as reference points.
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    def lin(i):
        sc = StandardScaler().fit(i[0])
        m = LogisticRegression(max_iter=3000).fit(sc.transform(i[0]), ys[0])
        return float(m.score(sc.transform(i[2]), ys[2]))

    info = dict(teacher=TEACHER, teacher_radius=ARGS.teacher_radius if TEACHER == "near" else None,
                teacher_seed=ARGS.teacher_seed, teacher_counts=counts.tolist(),
                teacher_rel_dist=float((theta - theta0).norm() / theta0.norm()),
                teacher_lin_descriptor=lin([f.numpy() for f in feats]),
                teacher_lin_angles=lin([a.numpy() for a in angles]),
                teacher_lin_readout=lin([q.numpy() for q in qs]))
    print("  Teacher labels: " + json.dumps(info))
    return ys, info


def run_epoch(model, loader, criterion, optimizer, device, train=True):
    model.train() if train else model.eval()
    total_loss, total_correct, total_n = 0.0, 0, 0
    preds_all, lbls_all = [], []
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for X, y in loader:
            X = X.to(device)
            y = y.to(device)
            if train:
                optimizer.zero_grad()
            out = model(X)
            loss = criterion(out, y)
            if train:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            total_loss += loss.item() * len(y)
            total_correct += (out.argmax(1) == y).sum().item()
            total_n += len(y)
            preds_all.extend(out.argmax(1).detach().cpu().numpy())
            lbls_all.extend(y.detach().cpu().numpy())
    bal = balanced_accuracy_score(lbls_all, preds_all)
    return total_loss / max(total_n, 1), total_correct / max(total_n, 1), bal


@torch.no_grad()
def _tta_views(imgs, mode):
    if mode == "flip":
        return [
            imgs,
            torch.flip(imgs, dims=[-1]),
            torch.flip(imgs, dims=[-2]),
            torch.flip(imgs, dims=[-2, -1]),
        ]
    if mode == "rot4":
        return [torch.rot90(imgs, k=k, dims=(-2, -1)) for k in range(4)]
    if mode == "d4":
        views = []
        for k in range(4):
            rot = torch.rot90(imgs, k=k, dims=(-2, -1))
            views.append(rot)
            views.append(torch.flip(rot, dims=[-1]))
        return views
    return [imgs]


@torch.no_grad()
def predict_logits(model, X, device, batch_size=64, tta_mode="none"):
    model.eval()
    ds = RSDataset(X, np.zeros(len(X), dtype=np.int64), aug=False)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    out = []
    for imgs, _ in loader:
        imgs = imgs.to(device)
        views = _tta_views(imgs, tta_mode)
        logits = None
        for view in views:
            cur = model(view)
            logits = cur if logits is None else logits + cur
        logits = logits / len(views)
        out.append(logits.cpu())
    return torch.cat(out, dim=0)


@torch.no_grad()
def predict(model, X, device, batch_size=64, tta_mode="none"):
    logits = predict_logits(model, X, device, batch_size=batch_size, tta_mode=tta_mode)
    return logits.argmax(1).numpy()


def main():
    meta = META[DS]
    tr_x, tr_y, val_x, val_y, te_x, te_y = load_data(DS)

    C = META[DS]["C"]
    K = meta["K"]
    size = meta["size"]
    names = meta["classes"]

    teacher_info = {}
    if TEACHER != "none":
        theta0 = replicate_theta0(torch.get_rng_state(), GlobalScatteringTransform(C, CONFIG["SCAT_J"]).out_dim)
        (tr_y, val_y, te_y), teacher_info = teacher_relabel([tr_x, val_x, te_x], K, theta0)
        names = [f"t{k}" for k in range(K)]

    model = SQ_SEQNN(C, size, K).to(DEVICE)
    params = model.count_params()
    if TEACHER != "none" and model.q_w is not None:
        assert torch.equal(model.q_w.detach().cpu(), theta0), "theta_0 replica differs from the model init"
        print("  theta_0 replica matches the model initialization")

    print("\n  Parameter Breakdown")
    print(f"    Scattering  : {params['scattering']:>5}")
    print(f"    RegionPool  : {params['region_pool']:>5}")
    print(f"    Compressor  : {params['compressor']:>5}")
    print(f"    Quantum QCNN: {params['quantum']:>5}")
    print(
        f"      ({N_REUP} re-upload rounds x {QCNN_PAIRS_PER_ROUND} odd/even pairs "
        f"x {QCNN_PAIR_PARAMS} params)"
    )
    print(f"    Classifier  : {params['classifier']:>5}")
    print(f"    {'-' * 24}")
    print(f"    TOTAL       : {params['total']:>5}")
    print(
        f"  Config: qubits={N_QUBITS}  basis={MEASURE_BASES}  "
        f"lr={CONFIG['LR']}  cls_hidden={CONFIG['CLS_HIDDEN']}  "
        f"frontend={FRONTEND_MODE}  gate={GATE_TYPE}"
        + (
            f"  scat_grid={CONFIG['SCAT_GRID']}  compress={COMPRESS_MODE}"
            if FRONTEND_MODE in {"local", "conv"} else ""
        )
    )
    if FRONTEND_MODE in {"local", "conv"}:
        extra = f"          summary={SUMMARY_MODE}  angle={ANGLE_MODE}"
        if FRONTEND_MODE == "conv":
            extra += f"  conv_width={CONFIG['CONV_WIDTH']}"
        print(extra)
    elif FRONTEND_MODE == "shapelet":
        print(
            f"          angle={ANGLE_MODE}  shapelets={CONFIG['SHAPELET_COUNT']}  "
            f"length={model.scat.length}  stats={CONFIG['SHAPELET_STATS']}"
        )
    else:
        print(f"          angle={ANGLE_MODE}")

    train_loader = balanced_loader(tr_x, tr_y, CONFIG["BATCH_SIZE"], aug=not ARGS.no_aug)
    val_loader = plain_loader(val_x, val_y, CONFIG["BATCH_SIZE"])

    criterion = nn.CrossEntropyLoss(label_smoothing=CONFIG["LABEL_SMOOTH"])
    hist = {"tr_l": [], "vl_l": [], "tr_a": [], "vl_a": [], "vl_b": []}
    stopper = EarlyStopping(CONFIG["PATIENCE"])

    dir_name = f"{DS}_{OVERHEAD_MODE if DS == 'overhead' else 'default'}"
    if RUN_TAG:
        dir_name = f"{dir_name}_{RUN_TAG}"
    out_dir = RESULTS_DIR / dir_name

    if EVAL_ONLY:
        model_path = MODEL_PATH or str(out_dir / f"sq_seqnn_{DS}.pt")
        print(f"\n  Eval-only mode: loading {model_path}")
        state = torch.load(model_path, map_location=DEVICE)
        model.load_state_dict(state)
    else:
        cls_params = [p for n, p in model.named_parameters() if n != "q_w"]
        q_params = [p for n, p in model.named_parameters() if n == "q_w" and p.requires_grad]
        param_groups = [{"params": cls_params, "lr": CONFIG["LR"]}]
        if q_params:
            param_groups.append(
                {"params": q_params, "lr": CONFIG["LR"] * CONFIG["LR_Q_FACTOR"]}
            )
        optimizer = optim.AdamW(param_groups, weight_decay=CONFIG["WEIGHT_DECAY"])
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=CONFIG["EPOCHS"], eta_min=1e-6
        )

        print(f"\n  Training for {CONFIG['EPOCHS']} epochs...")
        t0 = time.time()
        for ep in range(CONFIG["EPOCHS"]):
            tr_l, tr_a, _ = run_epoch(model, train_loader, criterion, optimizer, DEVICE, train=True)
            vl_l, vl_a, vl_b = run_epoch(model, val_loader, criterion, None, DEVICE, train=False)
            scheduler.step()
            hist["tr_l"].append(tr_l)
            hist["vl_l"].append(vl_l)
            hist["tr_a"].append(tr_a)
            hist["vl_a"].append(vl_a)
            hist["vl_b"].append(vl_b)
            print(
                f"  Ep {ep + 1:2d}/{CONFIG['EPOCHS']} | "
                f"tr {tr_l:.4f}/{tr_a:.3f} | "
                f"val {vl_l:.4f}/{vl_a:.3f} bal={vl_b:.3f}"
            )
            if stopper(vl_a, model):
                print(f"  Early stop at ep {ep + 1}  best={stopper.best:.3f}")
                break

        stopper.restore(model)
        print(f"\n  Done in {(time.time() - t0) / 60:.1f} min")

    print("\n" + "=" * 60)
    print(f"TEST RESULTS — {meta['name']}")
    print("=" * 60)

    y_pred = predict(model, te_x, DEVICE, tta_mode=TTA_MODE)
    acc = accuracy_score(te_y, y_pred)
    bal_acc = balanced_accuracy_score(te_y, y_pred)

    print(f"\n  Accuracy         : {acc:.4f}  ({acc * 100:.2f}%)")
    print(f"  Balanced Accuracy: {bal_acc:.4f}")
    print(f"  Total parameters : {params['total']}")
    print(f"  TTA mode         : {TTA_MODE}")
    print(f"\nPer-class report:")
    print(classification_report(te_y, y_pred, target_names=names, zero_division=0))

    if not EVAL_ONLY:
        out_dir.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), out_dir / f"sq_seqnn_{DS}.pt")

        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        ep_range = range(1, len(hist["tr_l"]) + 1)
        axes[0].plot(ep_range, hist["tr_l"], label="Train")
        axes[0].plot(ep_range, hist["vl_l"], label="Val")
        axes[0].set_title("Loss")
        axes[0].legend()
        axes[0].grid(alpha=0.3)
        axes[1].plot(ep_range, hist["tr_a"], label="Train")
        axes[1].plot(ep_range, hist["vl_a"], label="Val")
        axes[1].set_title("Accuracy")
        axes[1].legend()
        axes[1].grid(alpha=0.3)
        axes[2].plot(ep_range, hist["vl_b"], "g-")
        axes[2].axhline(1 / K, color="gray", linestyle="--", label="Chance")
        axes[2].set_ylim(0, 1)
        axes[2].set_title("Balanced Accuracy")
        axes[2].legend()
        axes[2].grid(alpha=0.3)
        plt.suptitle(f"SQ-SEQNN | {meta['name']} | {params['total']} params", fontsize=10)
        plt.tight_layout()
        plt.savefig(out_dir / f"training_{DS}.png", dpi=150)
        plt.close()

        cm = confusion_matrix(te_y, y_pred)
        fig, ax = plt.subplots(figsize=(max(8, K), max(7, K - 1)))
        ConfusionMatrixDisplay(cm, display_labels=names).plot(
            ax=ax, cmap="Blues", values_format="d", xticks_rotation=45
        )
        ax.set_title(f"SQ-SEQNN | {meta['name']} | {params['total']} params")
        plt.tight_layout()
        plt.savefig(out_dir / f"confusion_{DS}.png", dpi=150)
        plt.close()

    result_row = {
        "dataset": meta["name"],
        "overhead_mode": OVERHEAD_MODE if DS == "overhead" else "",
        "ablation": ABLATION,
        "no_fusion": NO_FUSION,
        "device": DEVICE.type,
        "quantum_mlp_params": params.get("quantum_mlp", 0),
        "accuracy": acc,
        "balanced_accuracy": bal_acc,
        "best_val_acc": None if EVAL_ONLY else stopper.best,
        "best_val_balanced_accuracy": None if not hist["vl_b"] else max(hist["vl_b"]),
        "total_params": params["total"],
        "quantum_params": N_Q_PARAMS,
        "n_qubits": N_QUBITS,
        "n_reuploading": N_REUP,
        "gate_type": GATE_TYPE,
        "measure_bases": MEASURE_BASES,
        "lr": CONFIG["LR"],
        "epochs": CONFIG["EPOCHS"],
        "batch_size": CONFIG["BATCH_SIZE"],
        "cls_hidden": CONFIG["CLS_HIDDEN"],
        "scat_grid": CONFIG["SCAT_GRID"],
        "conv_width": CONFIG["CONV_WIDTH"] if FRONTEND_MODE == "conv" else None,
        "shapelet_count": CONFIG["SHAPELET_COUNT"] if FRONTEND_MODE == "shapelet" else None,
        "shapelet_length": model.scat.length if FRONTEND_MODE == "shapelet" else None,
        "shapelet_stats": CONFIG["SHAPELET_STATS"] if FRONTEND_MODE == "shapelet" else None,
        "compress_mode": COMPRESS_MODE,
        "frontend_mode": FRONTEND_MODE,
        "summary_mode": SUMMARY_MODE,
        "angle_mode": ANGLE_MODE,
        "lr_q_factor": CONFIG["LR_Q_FACTOR"],
        "label_smooth": CONFIG["LABEL_SMOOTH"],
        "tta_mode": TTA_MODE,
        "eval_only": EVAL_ONLY,
        "quick": QUICK,
        "run_tag": RUN_TAG,
        "seed": CONFIG["SEED"],
        "no_aug": bool(ARGS.no_aug),
        **teacher_info,
    }
    if not EVAL_ONLY:
        pd.DataFrame([result_row]).to_csv(out_dir / f"results_{DS}.csv", index=False)
        with open(out_dir / f"results_{DS}.json", "w") as f:
            json.dump(result_row, f, indent=2)
        print(f"\n  Saved to {out_dir}")


if __name__ == "__main__":
    main()
