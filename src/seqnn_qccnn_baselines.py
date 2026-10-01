"""
SEQNN and QC-CNN baselines under the S2Q-SEQNN experimental protocol.

Purpose
-------
The original SEQNN and QC-CNN under the same splits as S2Q-SEQNN.
This module provides faithful
PyTorch + PennyLane reimplementations of both architectures, following
the published descriptions:

  QC-CNN : Fan et al., "Hybrid quantum-classical convolutional neural
           network model for image classification", IEEE TNNLS, 2023.
           Amplitude-encoded quantum convolution (quanvolution) over
           local image windows, followed by a classical head.

  SEQNN  : Fan et al., "Hybrid quantum deep learning with superpixel
           encoding for Earth observation data classification",
           IEEE TNNLS, 2025. Superpixel encoding feeding a VQC inside
           a squeeze-and-excitation recalibration pathway.

Honesty note for the manuscript / response letter
-------------------------------------------------
The original QC-CNN used TensorFlow Quantum and the official SEQNN code
lives at github.com/zhu-xlab/SEQNN. These are reimplementations in the
same framework, splits, seeds, optimizer, and training protocol as
S2Q-SEQNN, so the comparison is split-matched and pipeline-matched.
State this explicitly in the revision.

Environment notes (Apple Silicon)
---------------------------------
* Uses default.qubit (lightning.qubit segfaults on Apple Silicon).
* Uses diff_method="backprop", interface="torch".
* Quantum outputs are cast with .float() before .to(device).
* Classical layers can run on MPS; quantum circuits run on CPU.

Usage
-----
    from seqnn_qccnn_baselines import build_baseline, TrainConfig, run_seeds

    model_fn = lambda: build_baseline("seqnn", in_channels=1, num_classes=5)
    results  = run_seeds(model_fn, train_loader_fn, val_loader_fn,
                         test_loader_fn, TrainConfig(), seeds=range(42, 52))

Plug in your existing dataset loader functions from the S2Q-SEQNN repo
so the splits, normalisation (2nd-98th percentile), resizing (32x32),
and the weighted random sampler are byte-identical to the main model.
"""

from __future__ import annotations

import csv
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import pennylane as qml
import torch
import torch.nn as nn
import torch.nn.functional as F

# ----------------------------------------------------------------------
# Global config
# ----------------------------------------------------------------------

DEVICE = (
    torch.device("mps")
    if torch.backends.mps.is_available()
    else torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
)


@dataclass
class TrainConfig:
    """Matches the S2Q-SEQNN training protocol exactly."""

    epochs: int = 80
    batch_size: int = 64
    lr: float = 5e-3
    weight_decay: float = 1e-4
    label_smoothing: float = 0.05
    grad_clip: float = 1.0
    early_stop_patience: int = 20
    out_dir: str = "baseline_outputs"


# ======================================================================
# QC-CNN  (Fan et al., 2023)
# ======================================================================
#
# Core idea: a quantum convolution replaces one classical convolution.
# Each k x k window of the input is amplitude-encoded into
# ceil(log2(k*k*C_in)) qubits, processed by a shallow parameterised
# circuit, and the Pauli-Z expectations of the qubits form the output
# channels at that spatial position.
#
# Practical adaptation: quanvolution over the full 32x32 grid is
# computationally heavy on a state-vector simulator, so the quantum
# convolution is applied after an average-pool to a configurable
# resolution (default 8x8, giving 16 windows per image with a 2x2
# kernel and stride 2). This preserves the architectural role of the
# quantum layer while keeping ten-seed training tractable. The pooled
# resolution is a config knob, so a full-resolution run remains
# possible if compute allows.
# ----------------------------------------------------------------------


class QuanvLayer(nn.Module):
    """Amplitude-encoded quantum convolution (quanvolution).

    Windows of size (kernel x kernel x in_channels) are flattened,
    L2-normalised, amplitude-encoded on n_qubits, passed through
    `n_layers` of StronglyEntanglingLayers, and read out as Pauli-Z
    expectations on every qubit.
    """

    def __init__(self, in_channels: int, kernel: int = 2, stride: int = 2,
                 n_layers: int = 2):
        super().__init__()
        self.in_channels = in_channels
        self.kernel = kernel
        self.stride = stride

        window_dim = in_channels * kernel * kernel
        self.n_qubits = max(2, math.ceil(math.log2(window_dim)))
        self.pad_dim = 2 ** self.n_qubits

        dev = qml.device("default.qubit", wires=self.n_qubits)

        @qml.qnode(dev, interface="torch", diff_method="backprop")
        def circuit(amplitudes, weights):
            qml.AmplitudeEmbedding(
                amplitudes, wires=range(self.n_qubits), normalize=True
            )
            qml.StronglyEntanglingLayers(weights, wires=range(self.n_qubits))
            return [qml.expval(qml.PauliZ(w)) for w in range(self.n_qubits)]

        self.circuit = circuit
        shape = qml.StronglyEntanglingLayers.shape(
            n_layers=n_layers, n_wires=self.n_qubits
        )
        self.weights = nn.Parameter(0.1 * torch.randn(shape))
        self.out_channels = self.n_qubits

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W) on any device. Quantum part runs on CPU.
        b = x.shape[0]
        patches = F.unfold(x, kernel_size=self.kernel, stride=self.stride)
        # patches: (B, C*k*k, L) with L spatial positions
        n_pos = patches.shape[-1]
        patches = patches.permute(0, 2, 1).reshape(b * n_pos, -1).cpu()

        if patches.shape[-1] < self.pad_dim:
            pad = self.pad_dim - patches.shape[-1]
            patches = F.pad(patches, (0, pad))

        # All-black windows give zero-norm vectors, which AmplitudeEmbedding
        # cannot normalise (NaN). Encode empty patches as the |0...0> basis state.
        norms = patches.norm(dim=-1, keepdim=True)
        basis = torch.zeros_like(patches)
        basis[:, 0] = 1.0
        patches = torch.where(norms < 1e-8, basis, patches / norms.clamp_min(1e-9))

        outs = self.circuit(patches, self.weights.cpu())
        q = torch.stack(outs, dim=-1).float()          # (B*L, n_qubits)
        q = q.reshape(b, n_pos, self.out_channels)
        side = int(math.isqrt(n_pos))
        q = q.permute(0, 2, 1).reshape(b, self.out_channels, side, side)
        return q.to(x.device)


class QCCNN(nn.Module):
    """QC-CNN baseline: pooled input -> quanvolution -> classical head."""

    def __init__(self, in_channels: int, num_classes: int,
                 pooled_size: int = 8, hidden: int = 16):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(pooled_size)
        self.quanv = QuanvLayer(in_channels, kernel=2, stride=2)
        side = pooled_size // 2
        flat = self.quanv.out_channels * side * side
        self.head = nn.Sequential(
            nn.LayerNorm(flat),
            nn.Linear(flat, hidden),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, x):
        x = self.pool(x)
        q = self.quanv(x)
        return self.head(q.flatten(1))


# ======================================================================
# SEQNN  (Fan et al., 2025)
# ======================================================================
#
# Core idea: a compact convolutional backbone produces C feature maps.
# Superpixel encoding compresses the squeezed (globally averaged)
# channel descriptor into rotation angles for an n_q-qubit VQC. The
# circuit's Pauli-Z expectations pass through a small excitation head
# whose sigmoid output recalibrates the channels, exactly in the
# squeeze-and-excitation pattern, before the classifier.
# ----------------------------------------------------------------------


class SEQuantumExcitation(nn.Module):
    """VQC inside the squeeze-and-excitation pathway (Pauli-Z readout)."""

    def __init__(self, channels: int, n_qubits: int = 10, n_layers: int = 2):
        super().__init__()
        self.n_qubits = n_qubits
        self.squeeze_proj = nn.Linear(channels, n_qubits)

        dev = qml.device("default.qubit", wires=n_qubits)

        @qml.qnode(dev, interface="torch", diff_method="backprop")
        def circuit(angles, weights):
            for k in range(n_qubits):
                qml.RY(angles[..., k], wires=k)
            for k in range(n_qubits - 1):
                qml.CNOT(wires=[k, k + 1])
            qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
            return [qml.expval(qml.PauliZ(w)) for w in range(n_qubits)]

        self.circuit = circuit
        shape = qml.StronglyEntanglingLayers.shape(
            n_layers=n_layers, n_wires=n_qubits
        )
        self.weights = nn.Parameter(0.1 * torch.randn(shape))
        self.excite = nn.Linear(n_qubits, channels)

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        # feats: (B, C, H, W)
        squeezed = feats.mean(dim=(2, 3))                       # (B, C)
        angles = torch.tanh(self.squeeze_proj(squeezed)) * math.pi / 2
        outs = self.circuit(angles.cpu(), self.weights.cpu())
        q = torch.stack(outs, dim=-1).float().to(feats.device)  # (B, n_q)
        w = torch.sigmoid(self.excite(q)).unsqueeze(-1).unsqueeze(-1)
        return feats * w


class SEQNN(nn.Module):
    """SEQNN baseline: conv backbone + quantum SE recalibration + head."""

    def __init__(self, in_channels: int, num_classes: int,
                 width: int = 16, n_qubits: int = 10):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(in_channels, width, 3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(width, width, 3, stride=2, padding=1),
            nn.GELU(),
        )
        self.se = SEQuantumExcitation(width, n_qubits=n_qubits)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.LayerNorm(width),
            nn.Linear(width, num_classes),
        )

    def forward(self, x):
        f = self.backbone(x)
        f = self.se(f)
        return self.head(f)


# ======================================================================
# Factory, training harness, and ten-seed runner
# ======================================================================


def build_baseline(name: str, in_channels: int, num_classes: int) -> nn.Module:
    name = name.lower()
    if name == "qccnn":
        return QCCNN(in_channels, num_classes)
    if name == "seqnn":
        return SEQNN(in_channels, num_classes)
    raise ValueError(f"Unknown baseline '{name}', expected 'seqnn' or 'qccnn'.")


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


def train_one(model: nn.Module, train_loader, val_loader, cfg: TrainConfig):
    model = model.to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr,
                            weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs)
    crit = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)

    best_val, best_state, patience = 0.0, None, 0
    for epoch in range(cfg.epochs):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()
        sched.step()

        val_acc = evaluate(model, val_loader)
        if val_acc > best_val:
            best_val, patience = val_acc, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            patience += 1
            if patience >= cfg.early_stop_patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_val


@torch.no_grad()
def evaluate_balanced(model: nn.Module, loader) -> float:
    """Balanced accuracy: mean per-class recall, in percent."""
    model.eval()
    correct: dict = {}
    total: dict = {}
    for xb, yb in loader:
        pred = model(xb.to(DEVICE)).argmax(dim=-1).cpu()
        for p, y in zip(pred.tolist(), yb.tolist()):
            total[y] = total.get(y, 0) + 1
            correct[y] = correct.get(y, 0) + int(p == y)
    recalls = [100.0 * correct.get(c, 0) / n for c, n in total.items()]
    return sum(recalls) / max(len(recalls), 1)


@torch.no_grad()
def evaluate(model: nn.Module, loader) -> float:
    model.eval()
    correct = total = 0
    for xb, yb in loader:
        pred = model(xb.to(DEVICE)).argmax(dim=-1).cpu()
        correct += (pred == yb).sum().item()
        total += yb.numel()
    return 100.0 * correct / max(total, 1)


def run_seeds(
    model_fn: Callable[[], nn.Module],
    train_loader_fn: Callable[[int], object],
    val_loader_fn: Callable[[int], object],
    test_loader_fn: Callable[[int], object],
    cfg: TrainConfig,
    seeds: Iterable[int] = range(42, 52),
    tag: str = "baseline",
) -> dict:
    """Ten-seed protocol with mean, std, and 95 percent CI half-width.

    The loader functions take the seed so that the data split and the
    weighted sampler are seeded identically to the main S2Q-SEQNN runs.
    Results are appended to a CSV compatible with the existing
    ablation summary format.
    """
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"{tag}_runs.csv"

    accs = []
    t0 = time.time()
    for seed in seeds:
        set_seed(seed)
        model = model_fn()
        n_params = count_params(model)
        model, _ = train_one(model, train_loader_fn(seed), val_loader_fn(seed), cfg)
        acc = evaluate(model, test_loader_fn(seed))
        accs.append(acc)
        with open(csv_path, "a", newline="") as f:
            csv.writer(f).writerow([tag, seed, f"{acc:.4f}", n_params])
        print(f"[{tag}] seed {seed}: test acc {acc:.2f} ({n_params} params)")

    arr = np.asarray(accs)
    mean, std = arr.mean(), arr.std(ddof=1)
    ci95 = 1.96 * std / math.sqrt(len(arr))
    summary = {
        "tag": tag, "n_seeds": len(arr), "mean": mean, "std": std,
        "ci95_half": ci95, "minutes": (time.time() - t0) / 60.0,
    }
    print(f"[{tag}] {mean:.2f} +/- {std:.2f} (CI95 half {ci95:.2f})")
    return summary


if __name__ == "__main__":
    # Smoke test with random tensors, no real data.
    for name in ("seqnn", "qccnn"):
        m = build_baseline(name, in_channels=1, num_classes=5)
        x = torch.randn(4, 1, 32, 32)
        y = m(x)
        print(f"{name}: out {tuple(y.shape)}, params {count_params(m)}")
