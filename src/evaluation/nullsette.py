"""
Nullsette OOD (out-of-distribution) test for the Kosuri GPR pipeline.

Why this exists
---------------
The Kosuri 2013 "Nullsette" benchmark generates virtual-mutant expression
cassettes by *translocating* regulatory elements (promoter, RBS, terminator)
within the cassette. The resulting sequences are evolutionarily impossible
(no living cell would assemble them in this layout) but still contain the
same component sequences in a *different spatial arrangement*.

These Nullsettes have NO experimental ground truth, so they cannot be used
to compute R^2. Their job is to act as a *qualitative* OOD probe:

    1. Run a trained GPR on Normal (nonmutant.txt) cassettes.
    2. Run the same GPR on each Nullsette (mutant_translocation{1..19}.txt).
    3. Compare distributions of:
         - predicted mean expression
         - predicted standard deviation (the GP's own uncertainty)
    4. A model with *well-calibrated* uncertainty will show:
         - mean predictions similar between Normal and Nullsette (since the
           components are the same, biology is mostly preserved)
         - **std predictions significantly HIGHER for Nullsettes** (the model
           knows it has never seen this layout)

This file implements the full pipeline: FASTA parsing, fixed-window feature
extraction, prediction, distribution comparison (Mann-Whitney U + KS), and
plotting.

Usage
-----
    from src.evaluation.nullsette import run_nullsette_test
    summary = run_nullsette_test(
        model_path="src/runs/kosuri_gpr_1500_5fold__gpr/models/final_gpr.joblib",
        scaler_path="src/runs/kosuri_gpr_1500_5fold__gpr/models/scaler.joblib",
        kosuri_text_dir="data/processed/processed_data/kosuri",
        out_dir="src/runs/kosuri_gpr_1500_5fold__nullsette",
    )
"""

from __future__ import annotations

import glob
import os
import re
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu, ks_2samp

# Import the same feature helpers used by kosuri_loader
from src.data.kosuri_loader import extract_features as loader_extract_features


# ----------------------------------------------------------------------
# 1. FASTA parsing
# ----------------------------------------------------------------------
HEADER_RE = re.compile(r'^>"(?P<prom>[^"_]+)"_"(?P<rbs>[^"]+)"')


def parse_fasta(path: str) -> Dict[Tuple[str, str], str]:
    """Parse a FASTA-like file (">prom"_"rbs"\\n<seq>\\n>...").

    Returns: {(prom_id, rbs_id): sequence}
    Skips any record with a malformed header or empty sequence.
    """
    out: Dict[Tuple[str, str], str] = {}
    cur_key: Optional[Tuple[str, str]] = None
    cur_seq: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n").rstrip("\r")
            if line.startswith(">"):
                # commit the previous record
                if cur_key is not None and cur_seq:
                    out[cur_key] = "".join(cur_seq).upper()
                m = HEADER_RE.match(line)
                cur_key = (m.group("prom"), m.group("rbs")) if m else None
                cur_seq = []
            else:
                cur_seq.append(line)
        # final record
        if cur_key is not None and cur_seq:
            out[cur_key] = "".join(cur_seq).upper()
    return out


def load_nullsettes(kosuri_text_dir: str) -> Tuple[Dict, Dict[int, Dict]]:
    """Load all Nullsette files plus the non-mutant baseline.

    Returns
    -------
    normal : {(prom, rbs): seq}
        Map for the ``nonmutant.txt`` set (the real Kosuri cassettes).
    mutants : {n: dict}
        Map keyed by translocation index (1..19), each value is the same
        {(prom, rbs): seq} structure.
    """
    normal_path = os.path.join(kosuri_text_dir, "nonmutant.txt")
    if not os.path.exists(normal_path):
        raise FileNotFoundError(f"Missing {normal_path}")
    normal = parse_fasta(normal_path)

    mutants: Dict[int, Dict[Tuple[str, str], str]] = {}
    for path in sorted(glob.glob(os.path.join(kosuri_text_dir, "mutant_translocation*.txt"))):
        m = re.search(r"mutant_translocation(\d+)\.txt$", path)
        if not m:
            continue
        idx = int(m.group(1))
        mutants[idx] = parse_fasta(path)

    return normal, mutants


# ----------------------------------------------------------------------
# 2. Fixed-window feature extraction (no reliance on sd01/sd02 sequences)
# ----------------------------------------------------------------------
def split_cassette_by_atg(seq: str,
                          prom_window: int = 25,
                          rbs_window: int = 25) -> Tuple[Optional[str], Optional[str]]:
    """Find the FIRST ATG (CDS start codon) and carve out fixed windows before it.

    Kosuri cassettes do NOT contain the NdeI CATATG motif -- the RBS is
    assembled so the ATG start codon appears immediately after the
    Shine-Dalgarno-like region. We use the position of the first ATG as the
    anchor, then take a fixed-width RBS window right before it and a
    promoter window right before that.

    Returns (prom_subseq, rbs_subseq) or (None, None) if the ATG is too
    close to the start of the sequence for the windows to fit.
    """
    atg = seq.find("ATG")
    if atg < 0:
        return None, None
    if atg < (prom_window + rbs_window):
        return None, None
    rbs_subseq = seq[atg - rbs_window:atg]
    prom_subseq = seq[atg - prom_window - rbs_window:atg - rbs_window]
    return prom_subseq, rbs_subseq
def _feature_dict_from_subseq(prom_subseq: str, rbs_subseq: str) -> Dict[str, float]:
    """Compute the 20-feature dict used by the GPR model from raw subsequences.

    Mirrors the helpers inlined in src.data.kosuri_loader.extract_features.
    """
    def gc(s):
        return (s.count("G") + s.count("C")) / max(1, len(s))

    def at(s):
        return (s.count("A") + s.count("T")) / max(1, len(s))

    def skew(s):
        g, c = s.count("G"), s.count("C")
        return (g - c) / max(1, g + c)

    def complexity(s):
        if len(s) < 3:
            return 0.0
        return len(set(s[i:i+3] for i in range(len(s) - 2))) / (len(s) - 2)

    def mfe(s):
        return -gc(s) * 15.0

    out = {
        "prom__gc_content": gc(prom_subseq),
        "prom__at_content": at(prom_subseq),
        "prom__gc_skew": skew(prom_subseq),
        "prom__length": float(len(prom_subseq)),
        "prom__complexity": complexity(prom_subseq),
        "prom__mfe": mfe(prom_subseq),
        "rbs__gc_content": gc(rbs_subseq),
        "rbs__at_content": at(rbs_subseq),
        "rbs__gc_skew": skew(rbs_subseq),
        "rbs__length": float(len(rbs_subseq)),
        "rbs__complexity": complexity(rbs_subseq),
        "rbs__mfe": mfe(rbs_subseq),
        # Crude motif / SD scores on the windows (consistent with the model).
        "prom__score_minus35": (
            6.0 - sum(1 for a, b in zip(prom_subseq[:6], "TTGACA") if a != b)
        ) / 6.0 if len(prom_subseq) >= 6 else 0.0,
        "prom__score_minus10": (
            6.0 - sum(1 for a, b in zip(prom_subseq[10:16], "TATAAT") if a != b)
        ) / 6.0 if len(prom_subseq) >= 16 else 0.0,
        "prom__gc_motif_density": (
            prom_subseq.count("GC") + prom_subseq.count("CG")
        ) / max(1, len(prom_subseq)),
        "prom__at_motif_density": (
            prom_subseq.count("AT") + prom_subseq.count("TA")
        ) / max(1, len(prom_subseq)),
        "rbs__sd_best_score": (
            6.0 - sum(1 for a, b in zip(rbs_subseq[:6], "AGGAGG") if a != b)
        ) / 6.0 if len(rbs_subseq) >= 6 else 0.0,
        "rbs__sd_present": 1.0 if "AGGAG" in rbs_subseq else 0.0,
        "rbs__spacer_gc": gc(rbs_subseq[-7:]) if len(rbs_subseq) >= 7 else gc(rbs_subseq),
        "rbs__spacer_length": float(max(0, len(rbs_subseq) - 6)),
    }
    return out


def build_feature_table(record_dict: Dict[Tuple[str, str], str],
                        feature_cols: List[str]) -> pd.DataFrame:
    """Turn a {pair_key: full_seq} dict into a feature DataFrame aligned with feature_cols."""
    rows = []
    kept_keys = []
    for key, seq in record_dict.items():
        prom, rbs = split_cassette_by_atg(seq)
        if prom is None or rbs is None:
            continue
        d = _feature_dict_from_subseq(prom, rbs)
        d["promoter_name"] = key[0]
        d["rbs_name"] = key[1]
        rows.append(d)
        kept_keys.append(key)
    df = pd.DataFrame(rows)
    return df


# ----------------------------------------------------------------------
# 3. Predict + statistics
# ----------------------------------------------------------------------
def predict_with_uncertainty(model, scaler, df: pd.DataFrame,
                             feature_cols: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    """Run a trained (scaled) GPR on ``df[feature_cols]``.

    Returns (pred_mean, pred_std). If the model doesn't support return_std,
    pred_std will be zeros.
    """
    X = scaler.transform(df[feature_cols].values.astype(np.float64))
    try:
        mean, std = model.predict(X, return_std=True)
        return mean, std
    except TypeError:
        mean = model.predict(X)
        return mean, np.zeros_like(mean)


def compare_distributions(normal_pred: np.ndarray, normal_std: np.ndarray,
                          mut_pred: np.ndarray, mut_std: np.ndarray,
                          ) -> Dict[str, Dict[str, float]]:
    """Two-sided Mann-Whitney U + KS test for the mean and std predictions."""
    out = {}
    for name, a, b in [("pred_mean", normal_pred, mut_pred),
                       ("pred_std", normal_std, mut_std)]:
        mw_u, mw_p = mannwhitneyu(a, b, alternative="two-sided")
        ks_s, ks_p = ks_2samp(a, b)
        out[name] = {
            "normal_mean": float(np.mean(a)),
            "normal_std": float(np.std(a)),
            "mutant_mean": float(np.mean(b)),
            "mutant_std": float(np.std(b)),
            "mw_u": float(mw_u),
            "mw_pvalue": float(mw_p),
            "ks_stat": float(ks_s),
            "ks_pvalue": float(ks_p),
            "n_normal": int(len(a)),
            "n_mutant": int(len(b)),
        }
    return out


# ----------------------------------------------------------------------
# 4. Plotting
# ----------------------------------------------------------------------
def plot_distributions(normal_pred: np.ndarray, normal_std: np.ndarray,
                       mut_pred: np.ndarray, mut_std: np.ndarray,
                       out_path: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    bins = 40
    axes[0].hist(normal_pred, bins=bins, alpha=0.55, label=f"Normal (n={len(normal_pred)})", color="steelblue", density=True)
    axes[0].hist(mut_pred, bins=bins, alpha=0.55, label=f"Nullsette (n={len(mut_pred)})", color="coral", density=True)
    axes[0].set_xlabel("Predicted expression (log2)")
    axes[0].set_ylabel("Density")
    axes[0].set_title("Predicted MEAN distribution")
    axes[0].legend()

    axes[1].hist(normal_std, bins=bins, alpha=0.55, label=f"Normal (n={len(normal_std)})", color="steelblue", density=True)
    axes[1].hist(mut_std, bins=bins, alpha=0.55, label=f"Nullsette (n={len(mut_std)})", color="coral", density=True)
    axes[1].set_xlabel("Predicted std (GPR uncertainty)")
    axes[1].set_ylabel("Density")
    axes[1].set_title("Predicted STD distribution (the OOD signal)")
    axes[1].legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=130)
    plt.close()


# ----------------------------------------------------------------------
# 5. Top-level orchestrator
# ----------------------------------------------------------------------
def run_nullsette_test(model_path: str,
                       scaler_path: str,
                       kosuri_text_dir: str,
                       out_dir: str,
                       feature_cols: List[str]) -> Dict:
    """End-to-end OOD probe.

    1. Load trained model + scaler.
    2. Load Nullsette FASTA files.
    3. Extract features via fixed-window split (no leakage from sd01/sd02).
    4. Predict mean + std for each cassette.
    5. Compare distributions and write artefacts.
    """
    os.makedirs(out_dir, exist_ok=True)
    print(f"[NULLSETTE] loading model: {model_path}")
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)

    print(f"[NULLSETTE] loading Nullsettes from: {kosuri_text_dir}")
    normal, mutants = load_nullsettes(kosuri_text_dir)
    print(f"  normal records: {len(normal)}")
    print(f"  translocation variants: {len(mutants)} (indices {sorted(mutants.keys())})")

    # ----- Normal -----
    print(f"[NULLSETTE] building features for normal ...")
    df_normal = build_feature_table(normal, feature_cols)
    print(f"  usable normal rows: {len(df_normal)}")
    n_mean, n_std = predict_with_uncertainty(model, scaler, df_normal, feature_cols)

    # ----- Each mutant variant, then pool -----
    per_variant = []
    mut_mean_all = []
    mut_std_all = []
    for idx in sorted(mutants.keys()):
        df_m = build_feature_table(mutants[idx], feature_cols)
        if df_m.empty:
            print(f"  variant {idx:>2}: 0 rows (skipped)")
            continue
        m_mean, m_std = predict_with_uncertainty(model, scaler, df_m, feature_cols)
        per_variant.append({
            "variant": idx,
            "n": int(len(m_mean)),
            "mean_pred_mean": float(np.mean(m_mean)),
            "mean_pred_std": float(np.mean(m_std)),
            "std_pred_mean": float(np.std(m_mean)),
            "std_pred_std": float(np.std(m_std)),
        })
        mut_mean_all.append(m_mean)
        mut_std_all.append(m_std)
        print(f"  variant {idx:>2}: n={len(m_mean):>4}  pred_mean={np.mean(m_mean):.3f}  pred_std={np.mean(m_std):.3f}")

    mut_mean = np.concatenate(mut_mean_all) if mut_mean_all else np.array([])
    mut_std = np.concatenate(mut_std_all) if mut_std_all else np.array([])

    # ----- Distribution tests -----
    if len(mut_mean) == 0:
        raise RuntimeError("No mutant rows survived feature extraction; cannot run stats.")

    stats = compare_distributions(n_mean, n_std, mut_mean, mut_std)
    print("\n[NULLSETTE] distribution comparison:")
    for k, v in stats.items():
        print(f"  {k}: normal mean={v['normal_mean']:.3f} (std={v['normal_std']:.3f})  "
              f"mutant mean={v['mutant_mean']:.3f} (std={v['mutant_std']:.3f})  "
              f"KS p={v['ks_pvalue']:.2e}  MW p={v['mw_pvalue']:.2e}")

    # ----- Plot -----
    plot_path = os.path.join(out_dir, "ood_distribution.png")
    plot_distributions(n_mean, n_std, mut_mean, mut_std, plot_path)
    print(f"[NULLSETTE] plot -> {plot_path}")

    # ----- Per-variant CSV -----
    per_var_df = pd.DataFrame(per_variant)
    per_var_df.to_csv(os.path.join(out_dir, "per_variant_summary.csv"), index=False)

    # ----- Full prediction CSVs (handy for feature-importance follow-ups) -----
    df_normal[["promoter_name", "rbs_name"]].assign(pred_mean=n_mean, pred_std=n_std,
                                                   variant=0).to_csv(
        os.path.join(out_dir, "normal_predictions.csv"), index=False
    )

    # ----- Final summary -----
    summary = {
        "n_normal": int(len(n_mean)),
        "n_mutant_total": int(len(mut_mean)),
        "n_variants": len(per_variant),
        "distribution_tests": stats,
        "per_variant": per_variant,
        "model_path": model_path,
        "feature_cols": list(feature_cols),
    }
    import json
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"[NULLSETTE] summary -> {os.path.join(out_dir, 'summary.json')}")
    return summary



