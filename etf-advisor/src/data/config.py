"""Central configuration for the Phase 1 ETF data pipeline (Task 7).

Every constant that decides what the pipeline builds lives here, so the whole
dataset is reproducible from this one file. Nothing here performs I/O.
"""

from __future__ import annotations

from pathlib import Path

# --------------------------------------------------------------------------- #
# Universe
# --------------------------------------------------------------------------- #
UNIVERSE: list[str] = ["SPY", "QQQ", "EFA", "EEM", "AGG", "IEF", "GLD", "VNQ"]

# --------------------------------------------------------------------------- #
# Date ranges
# --------------------------------------------------------------------------- #
# Evaluation range: the target dates the model is ever trained/scored on.
EVAL_START = "2010-01-01"
EVAL_END = "2024-12-31"

# Fetch range: EVAL_START minus a warmup buffer. The pre-2010 span exists only
# so the longest lookback (50-day MA + 30-day window) is fully populated on the
# first evaluation date; samples whose window ends before EVAL_START are dropped.
FETCH_START = "2009-06-01"
FETCH_END = "2024-12-31"

# --------------------------------------------------------------------------- #
# Feature / target parameters
# --------------------------------------------------------------------------- #
WINDOW = 30  # trading days per input sequence

VOL_LOOKBACK = 10  # feature: rolling std of daily returns
MOM_SHORT, MOM_LONG = 5, 20  # feature: momentum horizons
MA_SHORT, MA_LONG = 20, 50  # feature: price / moving-average horizons

VOL_HORIZON = 20  # target: forward realised-vol horizon (trading days)

# Order matters: this is the column order of every window and of the scaler.
FEATURE_ORDER = ["ret", "vol_10", "mom_5", "mom_20", "px_ma20", "px_ma50"]

# --------------------------------------------------------------------------- #
# Splits (inclusive, keyed to sample label dates -- see windows.assign_splits)
# --------------------------------------------------------------------------- #
TRAIN_END = "2020-12-31"
VAL_START, VAL_END = "2021-01-01", "2022-12-31"
TEST_START, TEST_END = "2023-01-01", "2024-12-31"

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
# Anchored to the repo root (this file is src/data/config.py) so the pipeline
# and the notebooks resolve the same directories regardless of the CWD.
ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
CACHE_DIR = DATA_DIR / "cache"
PROCESSED_DIR = DATA_DIR / "processed"

# --------------------------------------------------------------------------- #
# Reproducibility
# --------------------------------------------------------------------------- #
SEED = 42
