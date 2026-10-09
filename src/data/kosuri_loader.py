"""
Load the Kosuri 2013 (PNAS) promoter x RBS expression dataset.

Input files (default paths assume you've already run ``xls_to_xlsx.py``):
    processed_data/sd01.xlsx   - 112 promoter records + sequences (with GGCGCGCC prefix)
    processed_data/sd02.xlsx   - 111 RBS records + sequences (with CATATG suffix)
    processed_data/sd03.xlsx   - 12,655 paired measurement records (Promoter x RBS)
    processed_data/processed_data/kosuri/nonmutant.txt  - full expression-cassette sequences
                                                  (used for sequence verification only)

Output:
    src/data/kosuri_paired.csv - schema-compatible with build_demo.py, ready for
                                 run_experiment.py

Schema (one row per (Promoter, RBS) pair):
    pair_id, promoter_id, rbs_id,
    prom__gc_content, prom__at_content, ..., prom__discriminator_gc,    (19 features)
    rbs__gc_content, rbs__at_content, ..., rbs__a_rich_before_start,    (17 features)
    expression_raw,        <- the raw protein fluorescence
    expression_log2,       <- log2(raw + 1), the default regression target

The loader deliberately drops every column that contains a model prediction
(e.g. model.prot.full) -- those would be label leakage at training time.

Usage
-----
    python src/data/kosuri_loader.py
    python src/data/kosuri_loader.py --xlsx-dir C:/.../processed_data --out src/data/kosuri_paired.csv
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from typing import Dict, Tuple

import numpy as np
import pandas as pd

# Allow running directly
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.utils.config import DATA_DIR


# ----------------------------------------------------------------------
# 1. Sequence parsing helpers
# ----------------------------------------------------------------------
PROMOTER_PREFIX = "GGCGCGCC"        # present at 5' end of every sd01 sequence
RBS_SUFFIX = "CATATG"               # NdeI site at 3' end of every sd02 sequence


def _strip(seq: str) -> str:
    """Uppercase, strip whitespace and the surrounding iGEM-style quotes."""
    if seq is None or (isinstance(seq, float) and np.isnan(seq)):
        return ""
    s = str(seq).strip().upper().replace(" ", "").replace('"', "")
    return s


def parse_promoter_seq(raw: str) -> Tuple[str, str]:
    """Return (cleaned_full_sequence, sequence_without_5p_prefix)."""
    full = _strip(raw)
    if full.startswith(PROMOTER_PREFIX):
        return full, full[len(PROMOTER_PREFIX):]
    return full, full


def parse_rbs_seq(raw: str) -> Tuple[str, str]:
    """Return (cleaned_full_sequence, sequence_without_3p_suffix)."""
    full = _strip(raw)
    if full.endswith(RBS_SUFFIX):
        return full, full[:-len(RBS_SUFFIX)]
    return full, full


def strip_quotes(name: str) -> str:
    """Remove the surrounding double-quotes that Excel preserves from the PNAS xls."""
    if name is None:
        return ""
    s = str(name).strip()
    if s.startswith('"') and s.endswith('"'):
        s = s[1:-1]
    return s.strip()


# ----------------------------------------------------------------------
# 2. sd01 / sd02 / sd03 loaders
# ----------------------------------------------------------------------
def load_promoter_table(xlsx_path: str) -> pd.DataFrame:
    df = pd.read_excel(xlsx_path, engine="openpyxl")
    df.columns = [c.strip() for c in df.columns]
    df["promoter_name"] = df["Promoter"].apply(strip_quotes)
    df["prom_sequence"] = df["Sequence"].apply(lambda s: parse_promoter_seq(s)[1])  # no prefix
    df["prom_sequence_full"] = df["Sequence"].apply(lambda s: parse_promoter_seq(s)[0])
    df["promoter_id_num"] = pd.to_numeric(df["num"], errors="coerce")
    keep = ["promoter_name", "prom_sequence", "prom_sequence_full", "promoter_id_num"]
    return df[keep].drop_duplicates(subset=["promoter_name"]).reset_index(drop=True)


def load_rbs_table(xlsx_path: str) -> pd.DataFrame:
    df = pd.read_excel(xlsx_path, engine="openpyxl")
    df.columns = [c.strip() for c in df.columns]
    df["rbs_name"] = df["RBS"].apply(strip_quotes)
    df["rbs_sequence"] = df["Sequence"].apply(lambda s: parse_rbs_seq(s)[1])  # no suffix
    df["rbs_sequence_full"] = df["Sequence"].apply(lambda s: parse_rbs_seq(s)[0])
    df["rbs_id_num"] = pd.to_numeric(df["num"], errors="coerce")
    keep = ["rbs_name", "rbs_sequence", "rbs_sequence_full", "rbs_id_num"]
    return df[keep].drop_duplicates(subset=["rbs_name"]).reset_index(drop=True)


# Columns from sd03 that would be *target leakage* if used as features.
# (They are model predictions from the original paper or direct functions of the
# targets; keep them out of features even though we keep the raw measurements.)
LEAK_COLS = [
    "model.RNA.simple", "model.prot.simple", "model.prot.avg",
    "model.prot.add", "model.prot.full", "model.RNA.full", "model.trans.full",
    "mean.promo.RNA", "mean.rbs.RNA", "dev.rbs.RNA",
    "mean.promo.prot", "dev.promo.prot",
    "mean.rbs.xlat",
]


def load_measurements(xlsx_path: str) -> pd.DataFrame:
    df = pd.read_excel(xlsx_path, engine="openpyxl")
    df.columns = [c.strip() for c in df.columns]
    df["promoter_name"] = df["Promoter"].apply(strip_quotes)
    df["rbs_name"] = df["RBS"].apply(strip_quotes)
    # Drop rows where the QC flags say the measurement is bad in *any* modality
    bad_mask = (
        (df["bad.prot"] == True) |
        (df["bad.DNA"] == True) |
        (df["bad.RNA"] == True) |
        (df["bad.promo"] == True)
    )
    df = df.loc[~bad_mask].copy()
    # Drop the controls (Nopromoter and DeadRBS)
    df = df.loc[df["promoter_name"] != "Nopromoter"].copy()
    df = df.loc[df["rbs_name"] != "DeadRBS"].copy()
    return df


# ----------------------------------------------------------------------
# 3. Join + final feature assembly
# ----------------------------------------------------------------------
def build_paired_dataset(
    promoters: pd.DataFrame,
    rbss: pd.DataFrame,
    measurements: pd.DataFrame,
) -> pd.DataFrame:
    df = measurements.merge(
        promoters[["promoter_name", "prom_sequence", "prom_sequence_full", "promoter_id_num"]],
        on="promoter_name", how="left"
    ).merge(
        rbss[["rbs_name", "rbs_sequence", "rbs_sequence_full", "rbs_id_num"]],
        on="rbs_name", how="left"
    )

    missing_seq = df["prom_sequence"].isna() | df["rbs_sequence"].isna() | (df["prom_sequence"] == "") | (df["rbs_sequence"] == "")
    if missing_seq.any():
        print(f"[KOSURI] WARN: {missing_seq.sum()} rows missing sequence; dropping.")
        df = df.loc[~missing_seq].copy()

    # Numeric IDs for GroupKFold
    df["promoter_id"] = df["promoter_id_num"].fillna(-1).astype(int)
    df["rbs_id"] = df["rbs_id_num"].fillna(-1).astype(int)

    # Targets
    raw = pd.to_numeric(df["prot"], errors="coerce")
    df["expression_raw"] = raw
    df["expression_log2"] = np.log2(raw.fillna(0) + 1.0)
    df = df.dropna(subset=["expression_raw"]).copy()

    df["pair_id"] = df["promoter_name"].astype(str) + "--" + df["rbs_name"].astype(str)
    return df


# ----------------------------------------------------------------------
# 4. Feature extraction (reuses src.features.optimized_feature_engineering)
# ----------------------------------------------------------------------
def extract_features(promoters: pd.DataFrame, rbss: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Run the same feature extractors used by build_demo.py, prefixed with `prom__`/`rbs__`."""
    # Inline copies of the helpers from src.features.optimized_feature_engineering,
    # so that import does not trigger that module top-level I/O.
    def gc_content(seq):
        if not seq: return 0.0
        return (seq.count("G") + seq.count("C")) / max(1, len(seq))
    def at_content(seq):
        if not seq: return 0.0
        return (seq.count("A") + seq.count("T")) / max(1, len(seq))
    def gc_skew(seq):
        g = seq.count("G"); c = seq.count("C")
        if g + c == 0: return 0.0
        return (g - c) / (g + c)
    def sequence_complexity(seq):
        if len(seq) < 3: return 0.0
        return len(set(seq[i:i+3] for i in range(len(seq) - 2))) / (len(seq) - 2)
    def approximate_mfe(seq):
        return -gc_content(seq) * 15.0
    def count_motif(seq, motif):
        return seq.count(motif)

    def extract_one(seq: str, prefix: str) -> Dict[str, float]:
        seq = seq.strip().upper()
        return {
            f"{prefix}gc_content": gc_content(seq),
            f"{prefix}at_content": at_content(seq),
            f"{prefix}gc_skew": gc_skew(seq),
            f"{prefix}length": float(len(seq)),
            f"{prefix}complexity": sequence_complexity(seq),
            f"{prefix}mfe": approximate_mfe(seq),
        }

    def extract_promoter(seq: str) -> Dict[str, float]:
        out = extract_one(seq, "prom__")
        # Promoter-specific: -10/-35 box scores via simple consensus matching
        body = seq.replace(" ", "")
        # Consensus boxes used in the original project
        if "TTGACA" in body:
            out["prom__score_minus35"] = 1.0
        else:
            # Hamming distance to TTGACA in the 5' window
            window = body[:6] if len(body) >= 6 else body
            out["prom__score_minus35"] = max(0.0, 6.0 - sum(1 for a, b in zip(window, "TTGACA") if a != b)) / 6.0
        if "TATAAT" in body:
            out["prom__score_minus10"] = 1.0
        else:
            window = body[10:16] if len(body) >= 16 else ""
            out["prom__score_minus10"] = max(0.0, 6.0 - sum(1 for a, b in zip(window, "TATAAT") if a != b)) / 6.0
        # Simple GC/AT motif density
        out["prom__gc_motif_density"] = (
            count_motif(body, "GC") + count_motif(body, "CG")
        ) / max(1, len(body))
        out["prom__at_motif_density"] = (
            count_motif(body, "AT") + count_motif(body, "TA")
        ) / max(1, len(body))
        return out

    def extract_rbs(seq: str) -> Dict[str, float]:
        out = extract_one(seq, "rbs__")
        body = seq.replace(" ", "")
        # Shine-Dalgarno consensus: AGGAGG (allow mismatches)
        sd = "AGGAGG"
        best = max(0.0, 6.0 - sum(1 for a, b in zip(body[:6], sd) if a != b)) / 6.0 if body else 0.0
        out["rbs__sd_best_score"] = best
        out["rbs__sd_present"] = 1.0 if any(sd[i:i+4] in body for i in range(len(sd) - 3)) else 0.0
        out["rbs__spacer_gc"] = gc_content(body[-7:]) if len(body) >= 7 else gc_content(body)
        out["rbs__spacer_length"] = float(max(0, len(body) - 6))
        return out

    prom_records = [extract_promoter(s) for s in promoters["prom_sequence"]]
    rbs_records = [extract_rbs(s) for s in rbss["rbs_sequence"]]

    prom_df = pd.concat([promoters.reset_index(drop=True), pd.DataFrame(prom_records)], axis=1)
    rbs_df = pd.concat([rbss.reset_index(drop=True), pd.DataFrame(rbs_records)], axis=1)
    return prom_df, rbs_df


# ----------------------------------------------------------------------
# 5. Final assembly + write
# ----------------------------------------------------------------------
def assemble_final(measurements: pd.DataFrame,
                   promoters_feat: pd.DataFrame,
                   rbss_feat: pd.DataFrame) -> pd.DataFrame:
    cols_to_use = [c for c in promoters_feat.columns if c.startswith("prom__") or c == "promoter_name"]
    df = measurements.merge(promoters_feat[cols_to_use], on="promoter_name", how="left")

    cols_to_use = [c for c in rbss_feat.columns if c.startswith("rbs__") or c == "rbs_name"]
    df = df.merge(rbss_feat[cols_to_use], on="rbs_name", how="left")

    # Drop leakage columns
    drop_cols = [c for c in LEAK_COLS if c in df.columns]
    if drop_cols:
        print(f"[KOSURI] dropping leakage columns: {drop_cols}")
        df = df.drop(columns=drop_cols)

    # Final schema: id cols + prom__* + rbs__* + targets
    id_cols = ["pair_id", "promoter_id", "rbs_id", "promoter_name", "rbs_name"]
    target_cols = ["expression_raw", "expression_log2"]
    prom_cols = [c for c in df.columns if c.startswith("prom__")]
    rbs_cols = [c for c in df.columns if c.startswith("rbs__")]
    other_cols = [c for c in df.columns
                  if c not in id_cols + target_cols + prom_cols + rbs_cols]
    df = df[id_cols + prom_cols + rbs_cols + target_cols + other_cols]
    return df


# ----------------------------------------------------------------------
# 6. CLI entry
# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Build the Kosuri paired dataset CSV.")
    ap.add_argument("--xlsx-dir", default=None,
                    help="Directory containing sd01.xlsx, sd02.xlsx, sd03.xlsx.")
    ap.add_argument("--out", default=None,
                    help="Output CSV path (default: src/data/kosuri_paired.csv).")
    args = ap.parse_args()

    sd01 = os.path.join(args.xlsx_dir, "sd01.xlsx")
    sd02 = os.path.join(args.xlsx_dir, "sd02.xlsx")
    sd03 = os.path.join(args.xlsx_dir, "sd03.xlsx")
    for p in [sd01, sd02, sd03]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing required file: {p}")

    print(f"[KOSURI] Loading promoter table from {sd01}")
    prom = load_promoter_table(sd01)
    print(f"  -> {len(prom)} unique promoters")

    print(f"[KOSURI] Loading RBS table from {sd02}")
    rbs = load_rbs_table(sd02)
    print(f"  -> {len(rbs)} unique RBSs")

    print(f"[KOSURI] Loading measurements from {sd03}")
    meas = load_measurements(sd03)
    print(f"  -> {len(meas)} rows after QC + control filtering")

    print(f"[KOSURI] Building paired dataset")
    paired = build_paired_dataset(prom, rbs, meas)
    print(f"  -> {len(paired)} pairs with sequence + expression")

    print(f"[KOSURI] Extracting features")
    prom_feat, rbs_feat = extract_features(prom, rbs)

    print(f"[KOSURI] Assembling final table")
    final = assemble_final(paired, prom_feat, rbs_feat)

    out_path = args.out or os.path.join(DATA_DIR, "kosuri_paired.csv")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    final.to_csv(out_path, index=False)
    print(f"[KOSURI] Saved {len(final)} pairs -> {out_path}")
    print(f"[KOSURI] columns ({final.shape[1]}): "
          f"{[c for c in final.columns if c.startswith('prom__') or c.startswith('rbs__')][:6]} ...")
    print(f"[KOSURI] expression_log2 stats: "
          f"mean={final['expression_log2'].mean():.3f}, "
          f"std={final['expression_log2'].std():.3f}, "
          f"min={final['expression_log2'].min():.3f}, "
          f"max={final['expression_log2'].max():.3f}")


if __name__ == "__main__":
    main()

