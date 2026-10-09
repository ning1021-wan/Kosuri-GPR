"""
Build a Kosuri-like paired dataset from the existing iGEM 48-promoter x 48-RBS tables.

Why this exists
---------------
The original iGEM-derived CSVs store promoter and RBS measurements independently
(48 + 48 rows). Kosuri 2013-style data, in contrast, stores *paired* promoter-RBS
combinations (12,563 rows) with a single expression measurement per pair.

To validate the new GroupKFold + config-driven pipeline before the real Kosuri data
arrives, this script synthesises a paired dataset by:

    1. Loading the 48 promoter features and 48 RBS features.
    2. Renaming columns with ``prom__`` / ``rbs__`` prefixes so the merge is
       unambiguous (both source frames contain columns like ``gc_content``).
    3. Computing a Cartesian product (48 x 48 = 2304 pairs).
    4. Defining the per-pair expression as the (log2) product of promoter and RBS
       strengths with multiplicative Gaussian noise:
           expression_raw = promoter_F_on_OD * 10^(rbs_log10) / SCALE
           expression_log2 = log2(expression_raw + eps) + noise
       This is a coarse but biologically plausible model: transcription rate
       (promoter) * translation rate (RBS) ~ steady-state protein level.

The output schema matches what the Kosuri loader will eventually produce:
    pair_id, promoter_id, rbs_id, <prom__*>, <rbs__*>,
    expression_raw, expression_log2

Usage
-----
    python src/data/build_demo.py
"""

import os
import sys
import numpy as np
import pandas as pd

# Allow running as `python src/data/build_demo.py` directly
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.utils.config import PROCESSED_DATA_DIR, DATA_DIR


# ----------------------------------------------------------------------
# 1. Load
# ----------------------------------------------------------------------
promoter_path = os.path.join(PROCESSED_DATA_DIR, "promoter_features_optimized.csv")
rbs_path = os.path.join(PROCESSED_DATA_DIR, "rbs_features_optimized.csv")

promoters_raw = pd.read_csv(promoter_path)
rbss_raw = pd.read_csv(rbs_path)
print(f"[LOAD] promoters: {promoters_raw.shape}, rbss: {rbss_raw.shape}")


# ----------------------------------------------------------------------
# 2. Define column groups (so we know which to prefix vs which to drop)
# ----------------------------------------------------------------------
ID_COLS = ["Strain"]                       # shared but harmless to prefix
PROM_STRENGTH_COL = "F_on_OD"
RBS_STRENGTH_COL = "F_on_OD_log"

# Columns that are *leakage* (direct functions of the targets or raw sequence)
LEAK_COLS = [
    "F_on_OD", "F_on_OD_log", "F-on/OD",
    "mean.RNA", "mean.prot",
    "Sequence", "seq_clean", "core_seq", "rbs_seq",
]

# Drop leakage + non-feature columns from each side BEFORE prefixing
NON_FEATURE_DROP = LEAK_COLS + ID_COLS


def prefix_columns(df: pd.DataFrame, prefix: str, drop_cols: list) -> pd.DataFrame:
    """Prefix every column with ``prefix`` (except those in ``drop_cols``)."""
    drop_cols = [c for c in drop_cols if c in df.columns]
    keep = [c for c in df.columns if c not in drop_cols]
    renamed = df[keep].rename(columns={c: f"{prefix}{c}" for c in keep})
    return renamed


promoters = prefix_columns(promoters_raw, "prom__", NON_FEATURE_DROP)
rbss = prefix_columns(rbss_raw, "rbs__", NON_FEATURE_DROP)

# Add numeric ids for grouping
promoters = promoters.reset_index(drop=False).rename(columns={"index": "promoter_id"})
rbss = rbss.reset_index(drop=False).rename(columns={"index": "rbs_id"})


# ----------------------------------------------------------------------
# 3. Cartesian product
# ----------------------------------------------------------------------
paired = promoters.merge(rbss, how="cross")
paired["pair_id"] = ["P{}_R{}".format(p, r) for p, r in
                     zip(paired["promoter_id"], paired["rbs_id"])]
print(f"[JOIN] paired rows: {paired.shape[0]}")


# ----------------------------------------------------------------------
# 4. Synthesise expression (log-space, multiplicative, noisy)
# ----------------------------------------------------------------------
rng = np.random.default_rng(42)

# Pull the original (non-prefixed) strength columns back from the raw tables
prom_strength = promoters_raw[PROM_STRENGTH_COL].values.astype(float)
rbs_strength_log10 = promoters_raw.iloc[:0].reindex(columns=[RBS_STRENGTH_COL]).columns  # placeholder
# We need per-row promoter strength by promoter_id; the Cartesian product means
# promoter_id appears once per RBS, so we can reindex the raw promoter DataFrame
# by integer id.
prom_strength_by_id = pd.Series(promoters_raw[PROM_STRENGTH_COL].values,
                                index=range(len(promoters_raw))).to_dict()
rbs_strength_by_id = pd.Series(rbss_raw[RBS_STRENGTH_COL].values,
                               index=range(len(rbss_raw))).to_dict()

prom_strength = np.array([prom_strength_by_id[p] for p in paired["promoter_id"]])
rbs_strength_log10 = np.array([rbs_strength_by_id[r] for r in paired["rbs_id"]])

# Convert RBS log10 -> linear
rbs_strength_lin = np.power(10.0, rbs_strength_log10)

# Multiplicative model: protein ~ transcription rate * translation rate
SCALE = 1.0e4
expression_raw = (prom_strength * rbs_strength_lin) / SCALE

# Multiplicative log-normal noise (sigma_log = 0.15 ~ 15% relative error)
sigma_log = 0.15
noise = rng.normal(loc=0.0, scale=sigma_log, size=len(paired))
expression_raw_noisy = expression_raw * np.exp(noise)

# Log2 target with a small epsilon for safety
EPS = 1.0
expression_log2 = np.log2(expression_raw_noisy + EPS)

paired["expression_raw"] = expression_raw_noisy
paired["expression_log2"] = expression_log2


# ----------------------------------------------------------------------
# 5. Reorder + save
# ----------------------------------------------------------------------
id_cols = ["pair_id", "promoter_id", "rbs_id"]
target_cols = ["expression_raw", "expression_log2"]
feature_cols = [c for c in paired.columns
                if c not in id_cols + target_cols]
paired = paired[id_cols + feature_cols + target_cols]

print(f"[CLEAN] final shape: {paired.shape}")
print(f"[CLEAN] n features: {len(feature_cols)}")

out_path = os.path.join(DATA_DIR, "paired_dataset.csv")
paired.to_csv(out_path, index=False)
print(f"[SAVE] {out_path}  ({paired.shape[0]} rows, {paired.shape[1]} cols)")

print("\n[HEAD]")
print(paired[id_cols + target_cols].head(5).to_string(index=False))
print(f"\n[STATS] expression_log2: mean={paired['expression_log2'].mean():.3f}, "
      f"std={paired['expression_log2'].std():.3f}, "
      f"min={paired['expression_log2'].min():.3f}, "
      f"max={paired['expression_log2'].max():.3f}")
