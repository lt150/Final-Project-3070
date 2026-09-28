"""Model, architecture grid, and target transforms for Task 8 Stages B/C.

Kept separate from the sweep script so Stage C can rebuild the selected
model from its checkpoint without importing any sweep logic.

Dropout interpretation (pre-registered before any results):
PyTorch's nn.LSTM applies dropout only BETWEEN stacked layers, so for the
grid's 1-layer rows (A2, A4) that placement is a no-op and the dropout
column would be meaningless. We therefore apply dropout in two places:
between stacked LSTM layers (native semantics — bites on A3 only) AND on
the last timestep's hidden state before the linear head (bites on every
arch). A1 has dropout 0.0, so it is unaffected either way.
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

N_FEATURES = 6
PRED_FLOOR = 1e-6  # T1 only: a linear head in raw space can go negative

# id -> (num_layers, hidden_size, dropout). Fixed grid, no per-run tuning.
ARCHS: dict[str, tuple[int, int, float]] = {
    "A1": (1, 32, 0.0),
    "A2": (1, 64, 0.2),
    "A3": (2, 64, 0.2),
    "A4": (1, 128, 0.3),
}

TARGETS = ("T1", "T2", "T3")


class LSTMVol(nn.Module):
    """Unidirectional LSTM over the 30x6 window -> dropout -> linear head."""

    def __init__(self, num_layers: int, hidden_size: int, dropout: float) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=N_FEATURES,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head_dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)                    # [B, L, H]
        last = self.head_dropout(out[:, -1, :])  # [B, H]
        return self.head(last).squeeze(-1)       # [B]


def build_model(arch_id: str) -> LSTMVol:
    layers, hidden, dropout = ARCHS[arch_id]
    return LSTMVol(layers, hidden, dropout)


def set_determinism(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")  # no-op on CPU
    torch.use_deterministic_algorithms(True)


@dataclass
class TargetSpec:
    """One pre-registered target: raw y_vol <-> standardised training space.

    ``mu``/``sd`` are train-only statistics of the target in its own space
    (z-standardisation, uniform recipe across T1/T2/T3). ``vol_trail_20`` is
    required by T3 only — it is the sample's trailing vol, sourced solely
    from baseline.parquet.
    """

    name: str
    mu: float
    sd: float

    def transform(
        self, y_vol: np.ndarray, vol_trail_20: np.ndarray | None = None
    ) -> np.ndarray:
        t = self._to_target_space(y_vol, vol_trail_20)
        return (t - self.mu) / self.sd

    def invert(
        self, z: np.ndarray, vol_trail_20: np.ndarray | None = None
    ) -> np.ndarray:
        """Standardised prediction -> raw daily vol units."""
        t = z * self.sd + self.mu
        if self.name == "T1":
            return np.maximum(t, PRED_FLOOR)
        if self.name == "T2":
            return np.exp(t)  # plain exponentiation: pre-registered Jensen stance
        if self.name == "T3":
            return vol_trail_20 * np.exp(t)
        raise ValueError(self.name)

    def _to_target_space(
        self, y_vol: np.ndarray, vol_trail_20: np.ndarray | None
    ) -> np.ndarray:
        if self.name == "T1":
            return y_vol
        if self.name == "T2":
            return np.log(y_vol)
        if self.name == "T3":
            return np.log(y_vol / vol_trail_20)
        raise ValueError(self.name)


def fit_target_spec(
    name: str, y_vol_train: np.ndarray, vol_trail_20_train: np.ndarray
) -> TargetSpec:
    """Fit mu/sd on TRAIN ONLY in the target's own space (float64)."""
    spec = TargetSpec(name=name, mu=0.0, sd=1.0)
    t = spec._to_target_space(
        y_vol_train.astype(np.float64), vol_trail_20_train.astype(np.float64)
    )
    return TargetSpec(name=name, mu=float(t.mean()), sd=float(t.std()))
