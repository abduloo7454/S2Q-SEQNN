#!/usr/bin/env python3
"""
Low-parameter classical baselines for the S2Q-SEQNN experiments.

Models:
  - TinyCNN
  - TinyResNet
  - TinyDenseNet
  - TinyViT
  - TinyMobileNet
  - TinyConvMixer
  - SpatialStatsLinear
  - SpatialStatsMLP

The goal is not to build strong ImageNet-scale baselines. The goal is to
compare S2Q-SEQNN against very small classical models under a similar
parameter-budget regime.

Example:
  MPLCONFIGDIR="$PWD/.mplconfig" \
  python src/lowparam_baselines.py --dataset overhead --model all --runs 3
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    f1_score,
)
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
BASE_DIR = Path(os.environ.get("S2Q_DATA_ROOT", REPO_ROOT / "data")).expanduser()

OVERHEAD_FULL10 = [
    "car", "harbor", "helicopter", "oil_gas_field", "parking_lot",
    "plane", "runway_mark", "ship", "stadium", "storage_tank",
]
OVERHEAD_FAIR5 = ["car", "harbor", "parking_lot", "plane", "ship"]

SAT6_CLASSES = ["building", "barren_land", "trees", "grassland", "road", "water"]
SO2SAT_CLASSES = [
    "urban_compact", "urban_open", "forest_trees", "low_vegetation", "bare_water"
]

SEN2_BANDS = [2, 1, 0, 6]  # B4, B3, B2, B8 -> RGBN
LCZ_TO_SEMANTIC = {
    0: 0, 1: 0, 2: 0,
    3: 1, 4: 1, 5: 1, 6: 1, 7: 1, 8: 1, 9: 1,
    10: 2, 11: 2, 12: 2,
    13: 3,
    14: 4, 15: 4, 16: 4,
}

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

MODEL_NAMES = [
    "cnn",
    "resnet",
    "densenet",
    "vit",
    "mobilenet",
    "convmixer",
    "stat_linear",
    "stat_mlp",
]
MATCHED_MODEL_NAMES = [name for name in MODEL_NAMES if name != "stat_linear"]
NATURAL_VARIANTS = {
    "cnn": "width=4, linear_head",
    "resnet": "width=4, linear_head",
    "densenet": "growth=3, layers=3",
    "vit": "dim=6, depth=1, heads=2",
    "mobilenet": "width=6",
    "convmixer": "dim=8, depth=1",
    "stat_linear": "parameter-free_stats + linear_head",
    "stat_mlp": "hidden=16",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train low-parameter CNN/ResNet/DenseNet/ViT baselines."
    )
    parser.add_argument(
        "--dataset",
        default="overhead",
        choices=["overhead", "sat6", "so2sat", "all"],
    )
    parser.add_argument(
        "--model",
        default="all",
        choices=MODEL_NAMES + ["matched", "all"],
        help="'matched' runs all trainable compact baselines except stat_linear.",
    )
    parser.add_argument(
        "--overhead-mode",
        default="seqnn5",
        choices=["seqnn5", "full10"],
        help="Use the same 5-class SEQNN-style split or all 10 classes.",
    )
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--label-smooth", type=float, default=0.05)
    parser.add_argument(
        "--min-params",
        type=int,
        default=1200,
        help="Preferred lower bound for matched-capacity baselines.",
    )
    parser.add_argument(
        "--max-params",
        type=int,
        default=1400,
        help="Preferred upper bound for matched-capacity baselines.",
    )
    parser.add_argument(
        "--arch-variant",
        default=None,
        help="Exact architecture variant string. Used by the parameter search runner.",
    )
    parser.add_argument(
        "--match-param-band",
        action="store_true",
        help="If set, choose variants inside --min-params and --max-params.",
    )
    parser.add_argument(
        "--out-dir",
        default=str(SCRIPT_DIR / "baseline_results" / "classical_lowparam_baselines"),
    )
    return parser


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        print("Device: Apple Silicon (MPS)")
        return torch.device("mps")
    if torch.cuda.is_available():
        print("Device: CUDA")
        return torch.device("cuda")
    print("Device: CPU")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def iqr_norm(arr: np.ndarray) -> np.ndarray:
    out = arr.copy().astype(np.float32)
    for c in range(arr.shape[1]):
        ch = arr[:, c, :, :]
        p2, p98 = np.percentile(ch, 2), np.percentile(ch, 98)
        ch = np.clip(ch, p2, p98)
        denom = p98 - p2
        out[:, c, :, :] = (ch - p2) / denom if denom > 0 else np.zeros_like(ch)
    return out


def sample_per_class(labels: np.ndarray, counts: dict[int, int], rng) -> np.ndarray:
    selected = []
    for cls, n_needed in counts.items():
        cls_idx = np.where(labels == cls)[0]
        if len(cls_idx) < n_needed:
            raise ValueError(f"Class {cls} has only {len(cls_idx)} samples.")
        selected.append(rng.choice(cls_idx, size=n_needed, replace=False))
    idx = np.concatenate(selected)
    rng.shuffle(idx)
    return idx


def cap_split_quick(x: np.ndarray, y: np.ndarray, max_total: int, seed: int):
    if len(y) <= max_total:
        return x, y
    keep_idx, _ = train_test_split(
        np.arange(len(y)),
        train_size=max_total,
        stratify=y,
        random_state=seed,
    )
    return x[keep_idx], y[keep_idx]


def load_overhead(args, seed: int):
    from PIL import Image

    classes = OVERHEAD_FULL10 if args.overhead_mode == "full10" else OVERHEAD_FAIR5
    meta = {
        "name": "Overhead-MNIST",
        "classes": classes,
        "K": len(classes),
        "C": 1,
        "size": 32,
    }

    def read_folder(base):
        ext = {".png", ".jpg", ".jpeg", ".bmp"}
        imgs, labels = [], []
        for ci, cn in enumerate(classes):
            folder = Path(base) / cn
            files = sorted([p for p in folder.iterdir() if p.suffix.lower() in ext])
            for p in files:
                img = Image.open(p).convert("L").resize((32, 32), Image.BILINEAR)
                imgs.append(np.asarray(img, dtype=np.float32)[None, :, :])
                labels.append(ci)
            print(f"  {cn:>15}: {sum(1 for y in labels if y == ci)}")
        return np.stack(imgs), np.asarray(labels, dtype=np.int64)

    print("[Overhead train source]")
    x_train_all, y_train_all = read_folder(PATHS["overhead"]["train"])
    print("[Overhead test source]")
    x_test_all, y_test_all = read_folder(PATHS["overhead"]["test"])

    if args.overhead_mode == "seqnn5":
        train_pool_total, val_total, test_total = 4443, 667, 557
        train_pool_idx, _ = train_test_split(
            np.arange(len(y_train_all)),
            train_size=train_pool_total,
            stratify=y_train_all,
            random_state=seed,
        )
        test_pool_idx, _ = train_test_split(
            np.arange(len(y_test_all)),
            train_size=test_total,
            stratify=y_test_all,
            random_state=seed,
        )
        x_pool, y_pool = x_train_all[train_pool_idx], y_train_all[train_pool_idx]
        x_test, y_test = x_test_all[test_pool_idx], y_test_all[test_pool_idx]
        val_ratio = val_total / train_pool_total
        x_train, x_val, y_train, y_val = train_test_split(
            x_pool,
            y_pool,
            test_size=val_ratio,
            stratify=y_pool,
            random_state=seed,
        )
    else:
        x_train, x_val, y_train, y_val = train_test_split(
            x_train_all,
            y_train_all,
            test_size=0.15,
            stratify=y_train_all,
            random_state=seed,
        )
        x_test, y_test = x_test_all, y_test_all

    if args.quick:
        x_train, y_train = cap_split_quick(x_train, y_train, 500, seed)
        x_val, y_val = cap_split_quick(x_val, y_val, 100, seed)
        x_test, y_test = cap_split_quick(x_test, y_test, 250, seed)

    return (
        iqr_norm(x_train),
        y_train,
        iqr_norm(x_val),
        y_val,
        iqr_norm(x_test),
        y_test,
        meta,
    )


def load_sat6(args, seed: int):
    import scipy.io as sio

    meta = {
        "name": "SAT-6",
        "classes": SAT6_CLASSES,
        "K": 6,
        "C": 4,
        "size": 32,
    }
    mat_path = PATHS["sat6"]["mat"]
    if not mat_path.exists():
        raise FileNotFoundError(f"SAT-6 file not found: {mat_path}")

    data = sio.loadmat(str(mat_path))
    rng = np.random.RandomState(seed)

    tr_x = data["train_x"].transpose(3, 2, 0, 1).astype(np.float32) / 255.0
    tr_y = np.argmax(data["train_y"].T, axis=1).astype(np.int64)
    te_x = data["test_x"].transpose(3, 2, 0, 1).astype(np.float32) / 255.0
    te_y = np.argmax(data["test_y"].T, axis=1).astype(np.int64)

    pad = 2
    tr_x = np.pad(tr_x, ((0, 0), (0, 0), (pad, pad), (pad, pad)), mode="constant")
    te_x = np.pad(te_x, ((0, 0), (0, 0), (pad, pad), (pad, pad)), mode="constant")

    train_pool_idx = sample_per_class(
        tr_y, {cls: 900 for cls in range(meta["K"])}, rng
    )
    test_idx = sample_per_class(te_y, {cls: 200 for cls in range(meta["K"])}, rng)

    pooled_x, pooled_y = tr_x[train_pool_idx], tr_y[train_pool_idx]
    x_train, x_val, y_train, y_val = train_test_split(
        pooled_x,
        pooled_y,
        test_size=200 / 900,
        stratify=pooled_y,
        random_state=seed,
    )
    x_test, y_test = te_x[test_idx], te_y[test_idx]

    if args.quick:
        x_train, y_train = cap_split_quick(x_train, y_train, 500, seed)
        x_val, y_val = cap_split_quick(x_val, y_val, 100, seed)
        x_test, y_test = cap_split_quick(x_test, y_test, 250, seed)

    return (
        iqr_norm(x_train),
        y_train,
        iqr_norm(x_val),
        y_val,
        iqr_norm(x_test),
        y_test,
        meta,
    )


def decode_h5_strings(values):
    arr = np.asarray(values)
    if arr.dtype.kind in {"S", "O"}:
        return np.array([
            v.decode("utf-8") if isinstance(v, (bytes, np.bytes_)) else str(v)
            for v in arr
        ])
    return arr.astype(str)


def load_so2sat(args, seed: int):
    import h5py

    meta = {
        "name": "So2Sat",
        "classes": SO2SAT_CLASSES,
        "K": 5,
        "C": 4,
        "size": 32,
    }

    def read_city_split(h5_path, geo_path, target_cities):
        with h5py.File(str(geo_path), "r") as gf:
            cities = decode_h5_strings(gf["city"][:])
        city_mask = np.isin(cities, target_cities)
        with h5py.File(str(h5_path), "r") as f:
            labels_oh = np.asarray(f["label"][city_mask], dtype=np.float32)
            sen2 = np.asarray(f["sen2"][city_mask], dtype=np.float32)

        lcz_raw = np.argmax(labels_oh, axis=1)
        mapped = np.array(
            [LCZ_TO_SEMANTIC.get(int(label), -1) for label in lcz_raw],
            dtype=np.int64,
        )
        valid = mapped >= 0
        mapped = mapped[valid]
        sen2 = sen2[valid]
        imgs = sen2[:, :, :, SEN2_BANDS].transpose(0, 3, 1, 2).astype(np.float32)
        for band_idx in range(imgs.shape[1]):
            band = imgs[:, band_idx]
            bmin, bmax = float(band.min()), float(band.max())
            imgs[:, band_idx] = (band - bmin) / (bmax - bmin + 1e-8)
        return imgs, mapped

    rng = np.random.RandomState(seed)
    imgs_bc, lbls_bc = read_city_split(
        PATHS["so2sat"]["train"], PATHS["so2sat"]["train_geo"], ["berlin", "cologne"]
    )
    imgs_mv, lbls_mv = read_city_split(
        PATHS["so2sat"]["val"], PATHS["so2sat"]["val_geo"], ["munich"]
    )
    imgs_mt, lbls_mt = read_city_split(
        PATHS["so2sat"]["test"], PATHS["so2sat"]["test_geo"], ["munich"]
    )

    all_imgs = np.concatenate([imgs_bc, imgs_mv, imgs_mt], axis=0)
    all_lbls = np.concatenate([lbls_bc, lbls_mv, lbls_mt], axis=0)
    sampled_idx = sample_per_class(
        all_lbls, {cls: 2000 for cls in range(meta["K"])}, rng
    )
    imgs_all, lbls_all = all_imgs[sampled_idx], all_lbls[sampled_idx]

    x_train, x_rest, y_train, y_rest = train_test_split(
        imgs_all,
        lbls_all,
        test_size=0.4,
        stratify=lbls_all,
        random_state=seed,
    )
    x_val, x_test, y_val, y_test = train_test_split(
        x_rest,
        y_rest,
        test_size=0.5,
        stratify=y_rest,
        random_state=seed,
    )

    if args.quick:
        x_train, y_train = cap_split_quick(x_train, y_train, 500, seed)
        x_val, y_val = cap_split_quick(x_val, y_val, 100, seed)
        x_test, y_test = cap_split_quick(x_test, y_test, 250, seed)

    return (
        iqr_norm(x_train),
        y_train,
        iqr_norm(x_val),
        y_val,
        iqr_norm(x_test),
        y_test,
        meta,
    )


def load_data(dataset: str, args, seed: int):
    if dataset == "overhead":
        return load_overhead(args, seed)
    if dataset == "sat6":
        return load_sat6(args, seed)
    if dataset == "so2sat":
        return load_so2sat(args, seed)
    raise ValueError(dataset)


def augment_np(img: np.ndarray) -> np.ndarray:
    out = img.copy()
    if np.random.rand() > 0.5:
        out = np.flip(out, axis=2).copy()
    if np.random.rand() > 0.5:
        out = np.flip(out, axis=1).copy()
    if np.random.rand() > 0.5:
        k = np.random.randint(0, 4)
        out = np.rot90(out, k, axes=(1, 2)).copy()
    if np.random.rand() > 0.7:
        out += np.random.normal(0, 0.015, out.shape).astype(np.float32)
    return np.clip(out, 0, 1).astype(np.float32)


class RSDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray, aug: bool = False):
        self.x = x
        self.y = torch.from_numpy(y).long()
        self.aug = aug

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        img = augment_np(self.x[idx]) if self.aug else self.x[idx]
        return torch.from_numpy(img.copy()).float(), self.y[idx]


def make_train_loader(x, y, batch_size: int, aug: bool):
    counts = np.bincount(y, minlength=int(y.max()) + 1)
    weights = (1.0 / np.maximum(counts, 1).astype(np.float32))[y]
    sampler = WeightedRandomSampler(torch.from_numpy(weights), len(y), replacement=True)
    return DataLoader(
        RSDataset(x, y, aug=aug),
        batch_size=batch_size,
        sampler=sampler,
        num_workers=0,
    )


def make_eval_loader(x, y, batch_size: int):
    return DataLoader(
        RSDataset(x, y, aug=False),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class ConvBNAct(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False),
            nn.GroupNorm(1, out_ch),
            nn.GELU(),
        )

    def forward(self, x):
        return self.net(x)


def classifier_head(in_dim: int, n_classes: int, hidden: Optional[int] = None):
    if hidden is None or hidden <= 0:
        return nn.Linear(in_dim, n_classes)
    return nn.Sequential(
        nn.LayerNorm(in_dim),
        nn.Linear(in_dim, hidden),
        nn.GELU(),
        nn.Linear(hidden, n_classes),
    )


class TinyCNN(nn.Module):
    def __init__(
        self,
        in_ch: int,
        n_classes: int,
        width: int = 4,
        head_hidden: Optional[int] = None,
    ):
        super().__init__()
        self.net = nn.Sequential(
            ConvBNAct(in_ch, width, stride=1),
            nn.AvgPool2d(2),
            ConvBNAct(width, width * 2, stride=1),
            nn.AvgPool2d(2),
            ConvBNAct(width * 2, width * 2, stride=1),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = classifier_head(width * 2, n_classes, head_hidden)

    def forward(self, x):
        x = self.net(x).flatten(1)
        return self.head(x)


class TinyResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = ConvBNAct(channels, channels)
        self.conv2 = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.GroupNorm(1, channels),
        )

    def forward(self, x):
        return F.gelu(x + self.conv2(self.conv1(x)))


class TinyResNet(nn.Module):
    def __init__(
        self,
        in_ch: int,
        n_classes: int,
        width: int = 4,
        head_hidden: Optional[int] = None,
    ):
        super().__init__()
        self.stem = ConvBNAct(in_ch, width)
        self.block1 = TinyResidualBlock(width)
        self.down = ConvBNAct(width, width * 2, stride=2)
        self.block2 = TinyResidualBlock(width * 2)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = classifier_head(width * 2, n_classes, head_hidden)

    def forward(self, x):
        x = self.stem(x)
        x = self.block1(x)
        x = self.down(x)
        x = self.block2(x)
        x = self.pool(x).flatten(1)
        return self.head(x)


class DenseLayer(nn.Module):
    def __init__(self, in_ch: int, growth: int):
        super().__init__()
        self.layer = ConvBNAct(in_ch, growth)

    def forward(self, x):
        return torch.cat([x, self.layer(x)], dim=1)


class TinyDenseNet(nn.Module):
    def __init__(self, in_ch: int, n_classes: int, growth: int = 3, layers: int = 3):
        super().__init__()
        self.stem = ConvBNAct(in_ch, growth)
        ch = growth
        dense_layers = []
        for _ in range(layers):
            dense_layers.append(DenseLayer(ch, growth))
            ch += growth
        self.features = nn.Sequential(*dense_layers)
        self.proj = ConvBNAct(ch, max(4, growth * 2), stride=2)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(max(4, growth * 2), n_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.features(x)
        x = self.proj(x)
        x = self.pool(x).flatten(1)
        return self.head(x)


class TinyTransformerBlock(nn.Module):
    def __init__(self, dim: int, heads: int = 2, mlp_ratio: float = 2.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=heads,
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, dim),
        )

    def forward(self, x):
        h = self.norm1(x)
        attn_out, _ = self.attn(h, h, h, need_weights=False)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x


class TinyViT(nn.Module):
    def __init__(
        self,
        in_ch: int,
        n_classes: int,
        image_size: int = 32,
        patch_size: int = 4,
        dim: int = 6,
        depth: int = 1,
        heads: int = 2,
    ):
        super().__init__()
        self.patch = nn.Conv2d(
            in_ch,
            dim,
            kernel_size=patch_size,
            stride=patch_size,
        )
        n_patches = (image_size // patch_size) ** 2
        self.cls = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos = nn.Parameter(torch.zeros(1, n_patches + 1, dim))
        self.blocks = nn.Sequential(*[
            TinyTransformerBlock(dim, heads=heads)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, n_classes)
        nn.init.trunc_normal_(self.pos, std=0.02)
        nn.init.trunc_normal_(self.cls, std=0.02)

    def forward(self, x):
        x = self.patch(x).flatten(2).transpose(1, 2)
        cls = self.cls.expand(x.shape[0], -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = x + self.pos[:, : x.shape[1]]
        x = self.blocks(x)
        return self.head(self.norm(x[:, 0]))


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(
                in_ch,
                in_ch,
                kernel_size=3,
                stride=stride,
                padding=1,
                groups=in_ch,
                bias=False,
            ),
            nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False),
            nn.GroupNorm(1, out_ch),
            nn.GELU(),
        )

    def forward(self, x):
        return self.net(x)


class TinyMobileNet(nn.Module):
    """Depthwise-separable CNN baseline with a very small parameter budget."""

    def __init__(self, in_ch: int, n_classes: int, width: int = 6):
        super().__init__()
        self.features = nn.Sequential(
            DepthwiseSeparableConv(in_ch, width),
            nn.AvgPool2d(2),
            DepthwiseSeparableConv(width, width * 2),
            nn.AvgPool2d(2),
            DepthwiseSeparableConv(width * 2, width * 2),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Linear(width * 2, n_classes)

    def forward(self, x):
        return self.head(self.features(x).flatten(1))


class ConvMixerBlock(nn.Module):
    def __init__(self, dim: int, kernel_size: int = 5):
        super().__init__()
        self.depthwise = nn.Sequential(
            nn.Conv2d(
                dim,
                dim,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
                groups=dim,
                bias=False,
            ),
            nn.GELU(),
            nn.GroupNorm(1, dim),
        )
        self.pointwise = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1, bias=False),
            nn.GELU(),
            nn.GroupNorm(1, dim),
        )

    def forward(self, x):
        x = x + self.depthwise(x)
        return self.pointwise(x)


class TinyConvMixer(nn.Module):
    """Patch-mixing classical baseline for comparison with tokenized S2Q features."""

    def __init__(
        self,
        in_ch: int,
        n_classes: int,
        patch_size: int = 4,
        dim: int = 8,
        depth: int = 1,
    ):
        super().__init__()
        self.patch = nn.Sequential(
            nn.Conv2d(in_ch, dim, kernel_size=patch_size, stride=patch_size),
            nn.GELU(),
            nn.GroupNorm(1, dim),
        )
        self.blocks = nn.Sequential(*[ConvMixerBlock(dim) for _ in range(depth)])
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(dim, n_classes)

    def forward(self, x):
        x = self.patch(x)
        x = self.blocks(x)
        x = self.pool(x).flatten(1)
        return self.head(x)


class SpatialStats(nn.Module):
    """
    Parameter-free spatial statistics per channel.

    Features per channel: mean, standard deviation, horizontal lag-1
    autocorrelation, vertical lag-1 autocorrelation, minimum, maximum.
    """

    def __init__(self, in_ch: int):
        super().__init__()
        self.out_dim = 6 * in_ch

    def forward(self, x):
        flat = x.flatten(2)
        mean = flat.mean(dim=2)
        std = flat.std(dim=2, unbiased=False)
        minv = flat.amin(dim=2)
        maxv = flat.amax(dim=2)

        centered = x - x.mean(dim=(2, 3), keepdim=True)
        left = centered[:, :, :, :-1]
        right = centered[:, :, :, 1:]
        top = centered[:, :, :-1, :]
        bottom = centered[:, :, 1:, :]

        h_num = (left * right).mean(dim=(2, 3))
        h_den = (
            left.std(dim=(2, 3), unbiased=False)
            * right.std(dim=(2, 3), unbiased=False)
        ).clamp_min(1e-6)
        v_num = (top * bottom).mean(dim=(2, 3))
        v_den = (
            top.std(dim=(2, 3), unbiased=False)
            * bottom.std(dim=(2, 3), unbiased=False)
        ).clamp_min(1e-6)

        hcorr = torch.nan_to_num(h_num / h_den, nan=0.0, posinf=0.0, neginf=0.0)
        vcorr = torch.nan_to_num(v_num / v_den, nan=0.0, posinf=0.0, neginf=0.0)
        return torch.cat([mean, std, hcorr, vcorr, minv, maxv], dim=1)


class SpatialStatsLinear(nn.Module):
    def __init__(self, in_ch: int, n_classes: int):
        super().__init__()
        self.stats = SpatialStats(in_ch)
        self.head = nn.Linear(self.stats.out_dim, n_classes)

    def forward(self, x):
        return self.head(self.stats(x))


class SpatialStatsMLP(nn.Module):
    def __init__(self, in_ch: int, n_classes: int, hidden: int = 16):
        super().__init__()
        self.stats = SpatialStats(in_ch)
        self.head = nn.Sequential(
            nn.LayerNorm(self.stats.out_dim),
            nn.Linear(self.stats.out_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, n_classes),
        )

    def forward(self, x):
        return self.head(self.stats(x))


def _with_variant(model: nn.Module, variant: str) -> nn.Module:
    model.baseline_variant = variant
    return model


def _select_model_candidate(
    candidates,
    min_params: int,
    max_params: int,
    variant_name: Optional[str] = None,
) -> nn.Module:
    built = []
    midpoint = 0.5 * (min_params + max_params)
    for variant, make_model in candidates:
        model = make_model()
        params = count_params(model)
        built.append((variant, model, params))

    if variant_name is not None:
        for variant, model, _ in built:
            if variant == variant_name:
                return _with_variant(model, variant)
        available = ", ".join(variant for variant, _, _ in built[:10])
        raise ValueError(
            f"Unknown variant '{variant_name}'. First available variants: {available}"
        )

    in_band = [item for item in built if min_params <= item[2] <= max_params]
    if in_band:
        variant, model, _ = sorted(in_band, key=lambda item: abs(item[2] - midpoint))[0]
        return _with_variant(model, variant)

    def distance_to_band(item):
        params = item[2]
        if params < min_params:
            return min_params - params
        return params - max_params

    variant, model, _ = sorted(built, key=lambda item: (distance_to_band(item), abs(item[2] - midpoint)))[0]
    return _with_variant(model, variant)


def build_model(
    name: str,
    in_ch: int,
    n_classes: int,
    image_size: int,
    min_params: int = 1200,
    max_params: int = 1400,
    variant_name: Optional[str] = None,
) -> nn.Module:
    if name == "cnn":
        candidates = []
        for width in range(3, 7):
            candidates.append((
                f"width={width}, linear_head",
                lambda width=width: TinyCNN(in_ch, n_classes, width=width),
            ))
            for hidden in range(6, 33, 2):
                candidates.append((
                    f"width={width}, head_hidden={hidden}",
                    lambda width=width, hidden=hidden: TinyCNN(
                        in_ch, n_classes, width=width, head_hidden=hidden
                    ),
                ))
        return _select_model_candidate(candidates, min_params, max_params, variant_name)
    if name == "resnet":
        candidates = []
        for width in range(2, 6):
            candidates.append((
                f"width={width}, linear_head",
                lambda width=width: TinyResNet(in_ch, n_classes, width=width),
            ))
            for hidden in range(4, 25):
                candidates.append((
                    f"width={width}, head_hidden={hidden}",
                    lambda width=width, hidden=hidden: TinyResNet(
                        in_ch, n_classes, width=width, head_hidden=hidden
                    ),
                ))
        return _select_model_candidate(candidates, min_params, max_params, variant_name)
    if name == "densenet":
        candidates = [
            (
                f"growth={growth}, layers={layers}",
                lambda growth=growth, layers=layers: TinyDenseNet(
                    in_ch, n_classes, growth=growth, layers=layers
                ),
            )
            for growth in range(2, 6)
            for layers in range(2, 6)
        ]
        return _select_model_candidate(candidates, min_params, max_params, variant_name)
    if name == "vit":
        candidates = []
        for dim in range(4, 13):
            for depth in range(1, 3):
                for heads in (1, 2, 3, 4):
                    if dim % heads == 0:
                        candidates.append((
                            f"dim={dim}, depth={depth}, heads={heads}",
                            lambda dim=dim, depth=depth, heads=heads: TinyViT(
                                in_ch,
                                n_classes,
                                image_size=image_size,
                                dim=dim,
                                depth=depth,
                                heads=heads,
                            ),
                        ))
        return _select_model_candidate(candidates, min_params, max_params, variant_name)
    if name == "mobilenet":
        candidates = [
            (
                f"width={width}",
                lambda width=width: TinyMobileNet(in_ch, n_classes, width=width),
            )
            for width in range(5, 15)
        ]
        return _select_model_candidate(candidates, min_params, max_params, variant_name)
    if name == "convmixer":
        candidates = [
            (
                f"dim={dim}, depth={depth}",
                lambda dim=dim, depth=depth: TinyConvMixer(
                    in_ch, n_classes, patch_size=4, dim=dim, depth=depth
                ),
            )
            for dim in range(5, 15)
            for depth in range(1, 4)
        ]
        return _select_model_candidate(candidates, min_params, max_params, variant_name)
    if name == "stat_linear":
        if variant_name not in (None, "parameter-free_stats + linear_head"):
            raise ValueError(f"Unknown stat_linear variant '{variant_name}'")
        return _with_variant(SpatialStatsLinear(in_ch, n_classes), "parameter-free_stats + linear_head")
    if name == "stat_mlp":
        candidates = [
            (
                f"hidden={hidden}",
                lambda hidden=hidden: SpatialStatsMLP(in_ch, n_classes, hidden=hidden),
            )
            for hidden in range(8, 129)
        ]
        return _select_model_candidate(candidates, min_params, max_params, variant_name)
    raise ValueError(name)


class EarlyStopper:
    def __init__(self, patience: int):
        self.patience = patience
        self.best = -1.0
        self.count = 0
        self.state = None

    def step(self, value: float, model: nn.Module) -> bool:
        if value > self.best + 1e-4:
            self.best = value
            self.count = 0
            self.state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            return False
        self.count += 1
        return self.count >= self.patience

    def restore(self, model: nn.Module, device: torch.device):
        if self.state is not None:
            model.load_state_dict({k: v.to(device) for k, v in self.state.items()})


def run_epoch(model, loader, criterion, optimizer, device):
    train = optimizer is not None
    model.train(train)
    total_loss, n = 0.0, 0
    preds, labels = [], []
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            if train:
                optimizer.zero_grad(set_to_none=True)
            out = model(x)
            loss = criterion(out, y)
            if train:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            total_loss += float(loss.item()) * len(y)
            n += len(y)
            preds.extend(out.argmax(1).detach().cpu().numpy())
            labels.extend(y.detach().cpu().numpy())
    acc = accuracy_score(labels, preds)
    bal = balanced_accuracy_score(labels, preds)
    macro_f1 = f1_score(labels, preds, average="macro")
    return total_loss / max(1, n), acc, bal, macro_f1


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    preds, labels = [], []
    for x, y in loader:
        out = model(x.to(device))
        preds.extend(out.argmax(1).cpu().numpy())
        labels.extend(y.numpy())
    return np.asarray(labels), np.asarray(preds)


def train_one(dataset: str, model_name: str, run_seed: int, args, device: torch.device):
    set_seed(run_seed)
    x_train, y_train, x_val, y_val, x_test, y_test, meta = load_data(dataset, args, run_seed)

    batch_size = args.batch_size or (32 if args.quick else 64)
    epochs = args.epochs if args.epochs is not None else (12 if args.quick else 120)
    patience = args.patience if args.patience is not None else (5 if args.quick else 18)

    train_loader = make_train_loader(x_train, y_train, batch_size, aug=True)
    val_loader = make_eval_loader(x_val, y_val, batch_size)
    test_loader = make_eval_loader(x_test, y_test, batch_size)

    requested_variant = args.arch_variant
    if requested_variant is None and not args.match_param_band:
        requested_variant = NATURAL_VARIANTS.get(model_name)

    model = build_model(
        model_name,
        meta["C"],
        meta["K"],
        meta["size"],
        min_params=args.min_params,
        max_params=args.max_params,
        variant_name=requested_variant,
    ).to(device)
    params = count_params(model)
    model_variant = getattr(model, "baseline_variant", "")
    if args.match_param_band:
        if params > args.max_params:
            print(f"  [WARN] {model_name} has {params} params > upper band {args.max_params}.")
        elif params < args.min_params and model_name != "stat_linear":
            print(f"  [WARN] {model_name} has {params} params < lower band {args.min_params}.")

    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smooth)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, epochs),
        eta_min=max(args.lr * 0.02, 1e-6),
    )
    stopper = EarlyStopper(patience)

    print("\n" + "=" * 72)
    print(f"Dataset={meta['name']} | Model={model_name} | seed={run_seed}")
    print(f"Train={len(y_train)} Val={len(y_val)} Test={len(y_test)} Params={params}")
    if args.match_param_band:
        print(f"Variant={model_variant} | target params={args.min_params}-{args.max_params}")
    else:
        print(f"Variant={model_variant} | natural compact baseline")
    print("=" * 72)

    start = time.time()
    history = []
    for ep in range(1, epochs + 1):
        tr_loss, tr_acc, tr_bal, tr_f1 = run_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc, val_bal, val_f1 = run_epoch(
            model, val_loader, criterion, None, device
        )
        scheduler.step()
        history.append({
            "epoch": ep,
            "train_loss": tr_loss,
            "train_accuracy": tr_acc,
            "train_balanced_accuracy": tr_bal,
            "train_macro_f1": tr_f1,
            "val_loss": val_loss,
            "val_accuracy": val_acc,
            "val_balanced_accuracy": val_bal,
            "val_macro_f1": val_f1,
        })
        if ep == 1 or ep % 5 == 0:
            print(
                f"  Ep {ep:3d}/{epochs} | "
                f"train={tr_acc:.4f} val={val_acc:.4f} "
                f"val_bal={val_bal:.4f}"
            )
        if stopper.step(val_acc, model):
            print(f"  Early stop at epoch {ep}. Best val={stopper.best:.4f}")
            break

    stopper.restore(model, device)
    elapsed_sec = time.time() - start

    train_loss, train_acc, train_bal, train_f1 = run_epoch(
        model, make_eval_loader(x_train, y_train, batch_size), criterion, None, device
    )
    val_loss, val_acc, val_bal, val_f1 = run_epoch(
        model, val_loader, criterion, None, device
    )
    test_loss, test_acc, test_bal, test_f1 = run_epoch(
        model, test_loader, criterion, None, device
    )
    y_true, y_pred = predict(model, test_loader, device)

    report = classification_report(
        y_true,
        y_pred,
        target_names=meta["classes"],
        zero_division=0,
        output_dict=True,
    )

    out = {
        "dataset": dataset,
        "dataset_name": meta["name"],
        "model": model_name,
        "model_variant": model_variant,
        "seed": run_seed,
        "params": params,
        "min_params": args.min_params,
        "max_params": args.max_params,
        "within_param_budget": params <= args.max_params,
        "within_param_band": args.min_params <= params <= args.max_params,
        "match_param_band": args.match_param_band,
        "train_loss": train_loss,
        "train_accuracy": train_acc,
        "train_balanced_accuracy": train_bal,
        "train_macro_f1": train_f1,
        "val_loss": val_loss,
        "val_accuracy": val_acc,
        "val_balanced_accuracy": val_bal,
        "val_macro_f1": val_f1,
        "test_loss": test_loss,
        "test_accuracy": test_acc,
        "test_balanced_accuracy": test_bal,
        "test_macro_f1": test_f1,
        "best_val_accuracy": stopper.best,
        "epochs_run": len(history),
        "elapsed_sec": elapsed_sec,
        "overhead_mode": args.overhead_mode if dataset == "overhead" else "",
        "quick": args.quick,
        "batch_size": batch_size,
        "lr": args.lr,
        "label_smooth": args.label_smooth,
    }

    print(
        f"  Result | train={train_acc:.4f} val={val_acc:.4f} "
        f"test={test_acc:.4f} bal={test_bal:.4f} f1={test_f1:.4f} "
        f"time={elapsed_sec/60:.1f} min"
    )

    return out, history, report, model.state_dict()


def aggregate_results(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    metric_cols = [
        "params",
        "train_loss",
        "train_accuracy",
        "train_balanced_accuracy",
        "train_macro_f1",
        "val_loss",
        "val_accuracy",
        "val_balanced_accuracy",
        "val_macro_f1",
        "test_loss",
        "test_accuracy",
        "test_balanced_accuracy",
        "test_macro_f1",
        "best_val_accuracy",
        "epochs_run",
        "elapsed_sec",
    ]
    agg_spec = {
        "n_runs": ("seed", "count"),
        "within_param_budget": ("within_param_budget", "all"),
        "within_param_band": ("within_param_band", "all"),
    }
    for col in metric_cols:
        agg_spec[f"mean_{col}"] = (col, "mean")
        agg_spec[f"std_{col}"] = (col, lambda s: float(s.std(ddof=0)))
    agg = df.groupby(
        ["dataset", "dataset_name", "model", "model_variant"],
        as_index=False,
    ).agg(**agg_spec)
    return agg.sort_values(
        ["dataset", "mean_test_accuracy", "mean_params"],
        ascending=[True, False, True],
    )


def save_outputs(out_dir: Path, rows, histories, reports, states):
    out_dir.mkdir(parents=True, exist_ok=True)
    runs_df = pd.DataFrame(rows)
    summary_df = aggregate_results(rows)
    runs_df.to_csv(out_dir / "baseline_runs.csv", index=False)
    summary_df.to_csv(out_dir / "baseline_summary.csv", index=False)
    with open(out_dir / "baseline_runs.json", "w") as f:
        json.dump(rows, f, indent=2)
    with open(out_dir / "baseline_summary.json", "w") as f:
        json.dump(summary_df.to_dict(orient="records"), f, indent=2)
    for key, hist in histories.items():
        pd.DataFrame(hist).to_csv(out_dir / f"history_{key}.csv", index=False)
    for key, report in reports.items():
        with open(out_dir / f"classification_report_{key}.json", "w") as f:
            json.dump(report, f, indent=2)
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)
    for key, state in states.items():
        torch.save(state, ckpt_dir / f"{key}.pt")

    separate_root = out_dir / "separate_by_dataset_model"
    for (dataset, model_name), group_df in runs_df.groupby(["dataset", "model"]):
        sub_dir = separate_root / dataset / model_name
        sub_dir.mkdir(parents=True, exist_ok=True)
        group_rows = group_df.to_dict(orient="records")
        group_summary = aggregate_results(group_rows)
        group_df.to_csv(sub_dir / "runs.csv", index=False)
        group_summary.to_csv(sub_dir / "summary.csv", index=False)
        with open(sub_dir / "runs.json", "w") as f:
            json.dump(group_rows, f, indent=2)
        with open(sub_dir / "summary.json", "w") as f:
            json.dump(group_summary.to_dict(orient="records"), f, indent=2)

        hist_dir = sub_dir / "histories"
        report_dir = sub_dir / "reports"
        model_dir = sub_dir / "checkpoints"
        hist_dir.mkdir(exist_ok=True)
        report_dir.mkdir(exist_ok=True)
        model_dir.mkdir(exist_ok=True)
        prefix = f"{dataset}_{model_name}_seed"
        for key, hist in histories.items():
            if key.startswith(prefix):
                pd.DataFrame(hist).to_csv(hist_dir / f"{key}.csv", index=False)
        for key, report in reports.items():
            if key.startswith(prefix):
                with open(report_dir / f"{key}.json", "w") as f:
                    json.dump(report, f, indent=2)
        for key, state in states.items():
            if key.startswith(prefix):
                torch.save(state, model_dir / f"{key}.pt")

    print(f"\nSaved baseline results to: {out_dir}")
    print(f"Separate folders: {out_dir / 'separate_by_dataset_model'}")
    print(summary_df.to_string(index=False))


def main():
    args = build_parser().parse_args()
    device = get_device()
    datasets = ["overhead", "sat6", "so2sat"] if args.dataset == "all" else [args.dataset]
    if args.model == "all":
        models = MODEL_NAMES
    elif args.model == "matched":
        models = MATCHED_MODEL_NAMES
    else:
        models = [args.model]
    out_dir = Path(args.out_dir)
    if args.quick:
        out_dir = out_dir / "quick"
    else:
        out_dir = out_dir / "full"

    all_rows = []
    histories = {}
    reports = {}
    states = {}

    for dataset in datasets:
        for model_name in models:
            for run_idx in range(args.runs):
                seed = args.seed + run_idx
                row, history, report, state = train_one(dataset, model_name, seed, args, device)
                key = f"{dataset}_{model_name}_seed{seed}"
                all_rows.append(row)
                histories[key] = history
                reports[key] = report
                states[key] = state
                save_outputs(out_dir, all_rows, histories, reports, states)


if __name__ == "__main__":
    main()
