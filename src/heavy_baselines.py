#!/usr/bin/env python3
"""
Classical baselines for S2Q-SEQNN.

The goal is to evaluate standard remote-sensing and computer-vision backbones
under the same dataset protocols used by the S2Q experiments. Models are
trained from scratch by default, with no ImageNet weights, so the comparison
tests parameter efficiency under the available data budget.

Supported heavy models:
  - resnet50
  - resnet101
  - vgg16
  - densenet121
  - efficientnet_b0
  - vit_b_16
  - swin_t

Supported compact models:
  - resnet
  - vgg
  - densenet
  - efficientnet
  - vit
  - swin

Example:
  MPLCONFIGDIR="$PWD/.mplconfig" \
  python src/heavy_baselines.py --dataset overhead --model resnet50 --runs 3 --epochs 80
"""

from __future__ import annotations

import argparse
import json
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
import torchvision.models as tvm
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    f1_score,
)

import lowparam_baselines as lb


SCRIPT_DIR = Path(__file__).resolve().parent

MODEL_NAMES = [
    "resnet50",
    "resnet101",
    "vgg16",
    "densenet121",
    "efficientnet_b0",
    "vit_b_16",
    "swin_t",
]

REDUCED_MODEL_NAMES = [
    "resnet",
    "vgg",
    "densenet",
    "efficientnet",
    "vit",
    "swin",
]
ALL_MODEL_NAMES = MODEL_NAMES + REDUCED_MODEL_NAMES

MODEL_DISPLAY_NAMES = {
    "resnet50": "ResNet-50",
    "resnet101": "ResNet-101",
    "vgg16": "VGG-16",
    "densenet121": "DenseNet-121",
    "efficientnet_b0": "EfficientNet-B0",
    "vit_b_16": "ViT-B/16",
    "swin_t": "Swin-Tiny",
    "resnet": "Compact ResNet",
    "vgg": "Compact VGG",
    "densenet": "Compact DenseNet",
    "efficientnet": "Compact EfficientNet",
    "vit": "Compact ViT",
    "swin": "Compact Swin",
}

REDUCED_TARGET_PARAMS = {
    "efficientnet": 1100,
    "resnet": 1200,
    "vit": 1375,
    "vgg": 1500,
    "swin": 1700,
    "densenet": 2000,
}


def display_model_name(model_name: str) -> str:
    return MODEL_DISPLAY_NAMES.get(model_name, model_name)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train heavy classical baselines on the S2Q dataset protocols."
    )
    parser.add_argument(
        "--dataset",
        default="overhead",
        choices=["overhead", "sat6", "so2sat", "all"],
    )
    parser.add_argument(
        "--model",
        default="all",
        choices=ALL_MODEL_NAMES + ["heavy", "reduced", "all"],
    )
    parser.add_argument("--overhead-mode", default="seqnn5", choices=["seqnn5", "full10"])
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--label-smooth", type=float, default=0.05)
    parser.add_argument(
        "--out-dir",
        default=str(SCRIPT_DIR / "baseline_results" / "heavy_classical_baselines"),
    )
    return parser


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        print("Device: Apple Silicon (MPS)")
        return torch.device("mps")
    if torch.cuda.is_available():
        print("Device: CUDA")
        return torch.device("cuda")
    print("Device: CPU")
    return torch.device("cpu")


class HeavyInputWrapper(nn.Module):
    """
    Resize small EO images and adapt arbitrary channel counts to 3 channels.

    This preserves the standard torchvision backbone structure while allowing
    Overhead-MNIST, SAT-6, and So2Sat inputs to share the same heavy baselines.
    """

    def __init__(self, backbone: nn.Module, in_ch: int, image_size: int):
        super().__init__()
        self.image_size = image_size
        if in_ch == 3:
            self.adapter = nn.Identity()
        else:
            self.adapter = nn.Conv2d(in_ch, 3, kernel_size=1)
        self.backbone = backbone

    def forward(self, x):
        x = self.adapter(x)
        if x.shape[-1] != self.image_size or x.shape[-2] != self.image_size:
            x = F.interpolate(
                x,
                size=(self.image_size, self.image_size),
                mode="bilinear",
                align_corners=False,
            )
        return self.backbone(x)


class ConvGNAct(nn.Module):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: int = 3,
        stride: int = 1,
        groups: int = 1,
    ):
        super().__init__()
        padding = kernel_size // 2
        self.net = nn.Sequential(
            nn.Conv2d(
                in_ch,
                out_ch,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=groups,
                bias=False,
            ),
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


class CompactResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = ConvGNAct(channels, channels)
        self.conv2 = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.GroupNorm(1, channels),
        )

    def forward(self, x):
        return F.gelu(x + self.conv2(self.conv1(x)))


class CompactResNet(nn.Module):
    def __init__(
        self,
        in_ch: int,
        n_classes: int,
        width: int = 3,
        head_hidden: int = 19,
    ):
        super().__init__()
        self.stem = ConvGNAct(in_ch, width)
        self.block1 = CompactResidualBlock(width)
        self.down = ConvGNAct(width, width * 2, stride=2)
        self.block2 = CompactResidualBlock(width * 2)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = classifier_head(width * 2, n_classes, head_hidden)

    def forward(self, x):
        x = self.stem(x)
        x = self.block1(x)
        x = self.down(x)
        x = self.block2(x)
        x = self.pool(x).flatten(1)
        return self.head(x)


class CompactVGG(nn.Module):
    def __init__(
        self,
        in_ch: int,
        n_classes: int,
        width: int = 5,
        head_hidden: int = 22,
    ):
        super().__init__()
        self.features = nn.Sequential(
            ConvGNAct(in_ch, width),
            ConvGNAct(width, width),
            nn.AvgPool2d(2),
            ConvGNAct(width, width * 2),
            ConvGNAct(width * 2, width * 2),
            nn.AvgPool2d(2),
            ConvGNAct(width * 2, width * 2),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = classifier_head(width * 2, n_classes, head_hidden)

    def forward(self, x):
        return self.head(self.features(x).flatten(1))


class CompactDenseLayer(nn.Module):
    def __init__(self, in_ch: int, growth: int):
        super().__init__()
        self.layer = ConvGNAct(in_ch, growth)

    def forward(self, x):
        return torch.cat([x, self.layer(x)], dim=1)


class CompactDenseNet(nn.Module):
    def __init__(
        self,
        in_ch: int,
        n_classes: int,
        growth: int = 4,
        layers: int = 2,
        head_hidden: int = 12,
    ):
        super().__init__()
        self.stem = ConvGNAct(in_ch, growth)
        ch = growth
        blocks = []
        for _ in range(layers):
            blocks.append(CompactDenseLayer(ch, growth))
            ch += growth
        self.blocks = nn.Sequential(*blocks)
        self.proj = ConvGNAct(ch, growth * 2, stride=2)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = classifier_head(growth * 2, n_classes, head_hidden)

    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        x = self.proj(x)
        x = self.pool(x).flatten(1)
        return self.head(x)


class CompactMBConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, expand: int = 2):
        super().__init__()
        mid = in_ch * expand
        self.use_skip = in_ch == out_ch
        self.net = nn.Sequential(
            ConvGNAct(in_ch, mid, kernel_size=1),
            ConvGNAct(mid, mid, kernel_size=3, groups=mid),
            nn.Conv2d(mid, out_ch, kernel_size=1, bias=False),
            nn.GroupNorm(1, out_ch),
        )

    def forward(self, x):
        y = self.net(x)
        if self.use_skip:
            y = y + x
        return F.gelu(y)


class CompactEfficientNet(nn.Module):
    def __init__(
        self,
        in_ch: int,
        n_classes: int,
        width: int = 7,
        head_hidden: int = 9,
    ):
        super().__init__()
        self.features = nn.Sequential(
            ConvGNAct(in_ch, width),
            nn.AvgPool2d(2),
            CompactMBConv(width, width),
            CompactMBConv(width, width * 2),
            nn.AvgPool2d(2),
            CompactMBConv(width * 2, width * 2),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = classifier_head(width * 2, n_classes, head_hidden)

    def forward(self, x):
        return self.head(self.features(x).flatten(1))


class CompactTransformerBlock(nn.Module):
    def __init__(self, dim: int, heads: int = 2, mlp_ratio: float = 2.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
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


class CompactViT(nn.Module):
    def __init__(
        self,
        in_ch: int,
        n_classes: int,
        image_size: int,
        patch_size: int = 4,
        dim: int = 8,
        depth: int = 1,
        heads: int = 2,
        head_hidden: int = 8,
    ):
        super().__init__()
        self.patch = nn.Conv2d(in_ch, dim, kernel_size=patch_size, stride=patch_size)
        n_patches = (image_size // patch_size) ** 2
        self.cls = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos = nn.Parameter(torch.zeros(1, n_patches + 1, dim))
        self.blocks = nn.Sequential(*[
            CompactTransformerBlock(dim, heads=heads)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(dim)
        self.head = classifier_head(dim, n_classes, head_hidden)
        nn.init.trunc_normal_(self.cls, std=0.02)
        nn.init.trunc_normal_(self.pos, std=0.02)

    def forward(self, x):
        x = self.patch(x).flatten(2).transpose(1, 2)
        cls = self.cls.expand(x.shape[0], -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = x + self.pos[:, : x.shape[1]]
        x = self.blocks(x)
        return self.head(self.norm(x[:, 0]))


class CompactSwinBlock(nn.Module):
    def __init__(self, dim: int, heads: int = 2):
        super().__init__()
        self.local = ConvGNAct(dim, dim, kernel_size=3, groups=dim)
        self.norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.mlp = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, dim),
        )

    def forward(self, x):
        x = x + self.local(x)
        b, c, h, w = x.shape
        tokens = x.flatten(2).transpose(1, 2)
        htok = self.norm(tokens)
        attn_out, _ = self.attn(htok, htok, htok, need_weights=False)
        tokens = tokens + attn_out
        tokens = tokens + self.mlp(tokens)
        return tokens.transpose(1, 2).reshape(b, c, h, w)


class CompactSwin(nn.Module):
    def __init__(
        self,
        in_ch: int,
        n_classes: int,
        dim: int = 8,
        heads: int = 2,
        head_hidden: int = 8,
    ):
        super().__init__()
        self.patch = ConvGNAct(in_ch, dim, kernel_size=4, stride=4)
        self.block = CompactSwinBlock(dim, heads=heads)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = classifier_head(dim, n_classes, head_hidden)

    def forward(self, x):
        x = self.patch(x)
        x = self.block(x)
        x = self.pool(x).flatten(1)
        return self.head(x)


def _with_variant(model: nn.Module, variant: str) -> nn.Module:
    model.baseline_variant = variant
    return model


def _select_reduced_candidate(
    candidates,
    target_params: int,
    min_params: int = 900,
    max_params: int = 2200,
) -> nn.Module:
    built = []
    for variant, make_model in candidates:
        model = make_model()
        params = count_params(model)
        built.append((variant, model, params))
    in_band = [item for item in built if min_params <= item[2] <= max_params]
    if in_band:
        variant, model, params = sorted(
            in_band, key=lambda item: abs(item[2] - target_params)
        )[0]
        return _with_variant(model, f"{variant}, target_params={target_params}, selected_params={params}")
    variant, model, params = sorted(
        built, key=lambda item: abs(item[2] - target_params)
    )[0]
    return _with_variant(model, f"{variant}, target_params={target_params}, selected_params={params}")


def build_reduced_model(
    model_name: str,
    in_ch: int,
    n_classes: int,
    image_size: int,
) -> nn.Module:
    target_params = REDUCED_TARGET_PARAMS[model_name]

    if model_name == "resnet":
        candidates = []
        for width in range(2, 6):
            for hidden in range(0, 81):
                candidates.append((
                    f"width={width}, head_hidden={hidden}",
                    lambda width=width, hidden=hidden: CompactResNet(
                        in_ch, n_classes, width=width, head_hidden=hidden
                    ),
                ))
        return _select_reduced_candidate(candidates, target_params)
    if model_name == "vgg":
        candidates = []
        for width in range(2, 7):
            for hidden in range(0, 81):
                candidates.append((
                    f"width={width}, head_hidden={hidden}",
                    lambda width=width, hidden=hidden: CompactVGG(
                        in_ch, n_classes, width=width, head_hidden=hidden
                    ),
                ))
        return _select_reduced_candidate(candidates, target_params)
    if model_name == "densenet":
        candidates = []
        for growth in range(2, 7):
            for layers in range(2, 6):
                for hidden in range(0, 81):
                    candidates.append((
                        f"growth={growth}, layers={layers}, head_hidden={hidden}",
                        lambda growth=growth, layers=layers, hidden=hidden: CompactDenseNet(
                            in_ch,
                            n_classes,
                            growth=growth,
                            layers=layers,
                            head_hidden=hidden,
                        ),
                    ))
        return _select_reduced_candidate(candidates, target_params)
    if model_name == "efficientnet":
        candidates = []
        for width in range(3, 10):
            for hidden in range(0, 81):
                candidates.append((
                    f"width={width}, head_hidden={hidden}",
                    lambda width=width, hidden=hidden: CompactEfficientNet(
                        in_ch, n_classes, width=width, head_hidden=hidden
                    ),
                ))
        return _select_reduced_candidate(candidates, target_params)
    if model_name == "vit":
        candidates = []
        for dim in range(5, 13):
            for depth in range(1, 3):
                for heads in (1, 2, 3, 4):
                    if dim % heads != 0:
                        continue
                    for hidden in range(0, 81):
                        candidates.append((
                            f"dim={dim}, depth={depth}, heads={heads}, head_hidden={hidden}",
                            lambda dim=dim, depth=depth, heads=heads, hidden=hidden: CompactViT(
                                in_ch,
                                n_classes,
                                image_size=image_size,
                                dim=dim,
                                depth=depth,
                                heads=heads,
                                head_hidden=hidden,
                            ),
                        ))
        return _select_reduced_candidate(candidates, target_params)
    if model_name == "swin":
        candidates = []
        for dim in range(6, 16):
            for heads in (1, 2, 3, 4):
                if dim % heads != 0:
                    continue
                for hidden in range(0, 81):
                    candidates.append((
                        f"dim={dim}, heads={heads}, head_hidden={hidden}",
                        lambda dim=dim, heads=heads, hidden=hidden: CompactSwin(
                            in_ch,
                            n_classes,
                            dim=dim,
                            heads=heads,
                            head_hidden=hidden,
                        ),
                    ))
        return _select_reduced_candidate(candidates, target_params)
    raise ValueError(model_name)


def replace_classifier(model_name: str, model: nn.Module, n_classes: int) -> nn.Module:
    if model_name in {"resnet50", "resnet101"}:
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, n_classes)
    elif model_name == "vgg16":
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, n_classes)
    elif model_name == "densenet121":
        in_features = model.classifier.in_features
        model.classifier = nn.Linear(in_features, n_classes)
    elif model_name == "efficientnet_b0":
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, n_classes)
    elif model_name == "vit_b_16":
        in_features = model.heads.head.in_features
        model.heads.head = nn.Linear(in_features, n_classes)
    elif model_name == "swin_t":
        in_features = model.head.in_features
        model.head = nn.Linear(in_features, n_classes)
    else:
        raise ValueError(model_name)
    return model


def build_backbone(model_name: str, n_classes: int) -> nn.Module:
    if model_name == "resnet50":
        model = tvm.resnet50(weights=None)
    elif model_name == "resnet101":
        model = tvm.resnet101(weights=None)
    elif model_name == "vgg16":
        model = tvm.vgg16(weights=None)
    elif model_name == "densenet121":
        model = tvm.densenet121(weights=None)
    elif model_name == "efficientnet_b0":
        model = tvm.efficientnet_b0(weights=None)
    elif model_name == "vit_b_16":
        model = tvm.vit_b_16(weights=None)
    elif model_name == "swin_t":
        model = tvm.swin_t(weights=None)
    else:
        raise ValueError(model_name)
    return replace_classifier(model_name, model, n_classes)


def build_model(model_name: str, in_ch: int, n_classes: int, image_size: int) -> nn.Module:
    if model_name in REDUCED_MODEL_NAMES:
        return build_reduced_model(model_name, in_ch, n_classes, image_size)
    backbone = build_backbone(model_name, n_classes)
    return HeavyInputWrapper(backbone, in_ch, image_size)


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

    def restore(self, model: nn.Module, device: torch.device) -> None:
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
    x_train, y_train, x_val, y_val, x_test, y_test, meta = lb.load_data(dataset, args, run_seed)
    model_label = display_model_name(model_name)

    if args.batch_size is not None:
        batch_size = args.batch_size
    elif args.quick:
        batch_size = 8
    else:
        batch_size = 8 if model_name in {"vgg16", "vit_b_16", "swin_t"} else 16
    patience = args.patience if args.patience is not None else (4 if args.quick else 15)

    train_loader = lb.make_train_loader(x_train, y_train, batch_size, aug=True)
    val_loader = lb.make_eval_loader(x_val, y_val, batch_size)
    test_loader = lb.make_eval_loader(x_test, y_test, batch_size)

    model = build_model(model_name, meta["C"], meta["K"], args.image_size).to(device)
    params = count_params(model)
    model_variant = getattr(model, "baseline_variant", "standard torchvision backbone")
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smooth)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, args.epochs),
        eta_min=max(args.lr * 0.02, 1e-6),
    )
    stopper = EarlyStopper(patience)

    print("\n" + "=" * 72)
    print(f"Dataset={meta['name']} | Baseline={model_label} | seed={run_seed}")
    print(
        f"Train={len(y_train)} Val={len(y_val)} Test={len(y_test)} "
        f"Params={params:,} Batch={batch_size} Image={args.image_size}"
    )
    print(f"Variant={model_variant}")
    print("=" * 72)

    start = time.time()
    history = []
    for ep in range(1, args.epochs + 1):
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
                f"  Ep {ep:3d}/{args.epochs} | "
                f"train={tr_acc:.4f} val={val_acc:.4f} val_bal={val_bal:.4f}"
            )
        if stopper.step(val_acc, model):
            print(f"  Early stop at epoch {ep}. Best val={stopper.best:.4f}")
            break

    stopper.restore(model, device)
    elapsed_sec = time.time() - start

    train_loss, train_acc, train_bal, train_f1 = run_epoch(
        model, lb.make_eval_loader(x_train, y_train, batch_size), criterion, None, device
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

    row = {
        "dataset": dataset,
        "dataset_name": meta["name"],
        "model_key": model_name,
        "model": model_label,
        "model_variant": model_variant,
        "seed": run_seed,
        "params": params,
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
        "batch_size": batch_size,
        "image_size": args.image_size,
        "lr": args.lr,
        "label_smooth": args.label_smooth,
        "overhead_mode": args.overhead_mode if dataset == "overhead" else "",
        "quick": args.quick,
    }
    print(
        f"  Result | train={train_acc:.4f} val={val_acc:.4f} "
        f"test={test_acc:.4f} bal={test_bal:.4f} f1={test_f1:.4f} "
        f"time={elapsed_sec/60:.1f} min"
    )
    return row, history, report, model.state_dict()


def aggregate_results(rows):
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
    agg_spec = {"n_runs": ("seed", "count")}
    for col in metric_cols:
        agg_spec[f"mean_{col}"] = (col, "mean")
        agg_spec[f"std_{col}"] = (col, lambda s: float(s.std(ddof=0)))
    out = df.groupby(["dataset", "dataset_name", "model", "model_variant"], as_index=False).agg(**agg_spec)
    return out.sort_values(
        ["dataset", "mean_test_accuracy", "mean_params"],
        ascending=[True, False, True],
    )


def write_markdown(df: pd.DataFrame, path: Path) -> None:
    cols = list(df.columns)
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
    ]
    for _, row in df.iterrows():
        values = [str(row[col]).replace("|", "/") for col in cols]
        lines.append("| " + " | ".join(values) + " |")
    path.write_text("\n".join(lines) + "\n")


def save_outputs(out_dir: Path, rows, histories, reports, states) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    runs_df = pd.DataFrame(rows)
    summary_df = aggregate_results(rows)
    runs_df.to_csv(out_dir / "heavy_baseline_runs.csv", index=False)
    summary_df.to_csv(out_dir / "heavy_baseline_summary.csv", index=False)
    with open(out_dir / "heavy_baseline_runs.json", "w") as f:
        json.dump(rows, f, indent=2)
    with open(out_dir / "heavy_baseline_summary.json", "w") as f:
        json.dump(summary_df.to_dict(orient="records"), f, indent=2)

    paper_rows = []
    for _, r in summary_df.iterrows():
        paper_rows.append({
            "Dataset": r["dataset_name"],
            "Baseline": r["model"],
            "Variant": r["model_variant"],
            "Params": int(round(r["mean_params"])),
            "Train Acc. (%)": f"{100*r['mean_train_accuracy']:.2f} ± {100*r['std_train_accuracy']:.2f}",
            "Val Acc. (%)": f"{100*r['mean_val_accuracy']:.2f} ± {100*r['std_val_accuracy']:.2f}",
            "Test Acc. (%)": f"{100*r['mean_test_accuracy']:.2f} ± {100*r['std_test_accuracy']:.2f}",
            "Balanced Acc. (%)": f"{100*r['mean_test_balanced_accuracy']:.2f} ± {100*r['std_test_balanced_accuracy']:.2f}",
            "Macro-F1 (%)": f"{100*r['mean_test_macro_f1']:.2f} ± {100*r['std_test_macro_f1']:.2f}",
            "Time/Run (s)": f"{r['mean_elapsed_sec']:.1f} ± {r['std_elapsed_sec']:.1f}",
        })
    paper_df = pd.DataFrame(paper_rows)
    paper_df.to_csv(out_dir / "heavy_baseline_paper_table.csv", index=False)
    write_markdown(paper_df, out_dir / "heavy_baseline_paper_table.md")
    with open(out_dir / "heavy_baseline_paper_table.tex", "w") as f:
        f.write(paper_df.to_latex(index=False, escape=False))

    for key, hist in histories.items():
        pd.DataFrame(hist).to_csv(out_dir / f"history_{key}.csv", index=False)
    for key, report in reports.items():
        with open(out_dir / f"classification_report_{key}.json", "w") as f:
            json.dump(report, f, indent=2)
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)
    for key, state in states.items():
        torch.save(state, ckpt_dir / f"{key}.pt")

    print(f"\nSaved heavy baseline results to: {out_dir}")
    print(summary_df.to_string(index=False))


def main():
    args = build_parser().parse_args()
    device = get_device()
    datasets = ["overhead", "sat6", "so2sat"] if args.dataset == "all" else [args.dataset]
    if args.model == "all":
        models = ALL_MODEL_NAMES
    elif args.model == "heavy":
        models = MODEL_NAMES
    elif args.model == "reduced":
        models = REDUCED_MODEL_NAMES
    else:
        models = [args.model]
    out_dir = Path(args.out_dir) / ("quick" if args.quick else "full")

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
