"""
Unified path + configuration registry for the Kosuri-GPR-Seq2Expr pipeline.

This file is the single source of truth for *where things live*. All hyperparameters
(features, models, CV splits) live in JSON configs under CONFIG_DIR.

NOTE on sandbox layout
----------------------
The sandbox only grants write access to ``src/``. Therefore all new artefacts
(output CSVs, run logs, plots, persisted models) are written under ``src/``
sub-directories, while the original iGEM tables (read-only) remain in
``<project_root>/data/processed``.
"""

import os

# ============================================================
# Project root + key directories
# ============================================================
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

SRC_DIR = os.path.join(PROJECT_ROOT, 'src')
CONFIG_DIR = os.path.join(SRC_DIR, 'configs')
DATA_DIR = os.path.join(SRC_DIR, 'data')                 # writable
RUNS_DIR = os.path.join(SRC_DIR, 'runs')                 # writable
ARTIFACTS_DIR = os.path.join(SRC_DIR, 'artifacts')       # writable

# Read-only source data (iGEM tables, raw sequences)
RAW_DATA_DIR = os.path.join(PROJECT_ROOT, 'data', 'raw')
PROCESSED_DATA_DIR = os.path.join(PROJECT_ROOT, 'data', 'processed')

# Legacy results dirs (read-only in sandbox; we will still create inside src/)
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results')
FEATURE_RESULTS_DIR = os.path.join(RESULTS_DIR, 'feature_results')
GP_RESULTS_DIR = os.path.join(RESULTS_DIR, 'gp_results')
AL_RESULTS_DIR = os.path.join(RESULTS_DIR, 'active_learning')

# Models + plots go inside src/artifacts (sandbox-friendly)
MODEL_DIR = os.path.join(ARTIFACTS_DIR, 'models')
PLOT_DIR = os.path.join(ARTIFACTS_DIR, 'plots')
RUN_PLOT_DIR = os.path.join(RUNS_DIR, 'plots')
RUN_MODEL_DIR = os.path.join(RUNS_DIR, 'models')

# ============================================================
# Ensure all writable directories exist
# ============================================================
for d in [
    CONFIG_DIR, DATA_DIR, RUNS_DIR, ARTIFACTS_DIR,
    MODEL_DIR, PLOT_DIR, RUN_PLOT_DIR, RUN_MODEL_DIR
]:
    os.makedirs(d, exist_ok=True)

# ============================================================
# Legacy aliases (kept for back-compat with old scripts)
# ============================================================
RAW_DIR = RAW_DATA_DIR
PROCESSED_DIR = PROCESSED_DATA_DIR
FEATURE_DIR = FEATURE_RESULTS_DIR
GP_DIR = GP_RESULTS_DIR
MODEL_DIR_FINAL = MODEL_DIR
PLOT_DIR_FINAL = PLOT_DIR


# ============================================================
# Config-file path helpers
# ============================================================
def config_path(name):
    """Resolve a JSON config name to an absolute path.

    Example:
        >>> config_path("model")
        'C:/.../src/configs/model.json'
    """
    if not name.endswith(".json"):
        name = name + ".json"
    return os.path.join(CONFIG_DIR, name)
