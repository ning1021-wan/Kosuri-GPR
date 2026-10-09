"""
GPR Training - Fixed Version
Problem: normalize_y=True + StandardScaler conflict
Solution: Remove normalize_y, manually standardize X only
"""

import sys
import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import joblib
import warnings
warnings.filterwarnings('ignore')

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)
from src.utils.config import PROCESSED_DATA_DIR

OUTPUT_DIR = os.path.join(project_root, 'results', 'gp_results')
MODEL_DIR = os.path.join(OUTPUT_DIR, 'models')
PLOT_DIR = os.path.join(OUTPUT_DIR, 'plots')
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)

print("=" * 70)
print("GPR TRAINING - FIXED VERSION")
print("=" * 70)

# ============================================================
# 1. LOAD DATA
# ============================================================
promoters = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, 'promoter_features_optimized.csv'))
rbss = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, 'rbs_features_optimized.csv'))

print(f"Promoters: {len(promoters)}, RBS: {len(rbss)}")

# ============================================================
# 2. FEATURES (SAME AS BEFORE)
# ============================================================
promoter_features = ['gc_motif_density', 'score_minus10']     
# 'dist_minus10', 'at_content', 'up_at_content', 'gc_content'
promoter_target = 'F_on_OD'

rbs_features = ['sd_best_score', 'spacer_gc',]
# 'sd_sequence_score', 'sd_end_pos', 'sd_start_pos', 'sd_mismatches'
rbs_target = 'F_on_OD_log'

print("\nPromoter Features:", promoter_features)
print("RBS Features:", rbs_features)
print(f"RBS Target: {rbs_target}")

# ============================================================
# 3. PREPARE DATA - FIXED: Only scale X, NOT y
# ============================================================
def prepare_data(df, features, target):
    X = np.nan_to_num(df[features].values)
    y = np.nan_to_num(df[target].values).reshape(-1, 1)  # Keep as 2D for scaling check
    
    # Scale X only
    scaler_X = StandardScaler()
    X_scaled = scaler_X.fit_transform(X)
    
    # Do NOT scale y - GPR handles it internally
    
    return X_scaled, y.ravel(), scaler_X

X_prom, y_prom, scaler_prom = prepare_data(promoters, promoter_features, promoter_target)
X_rbs, y_rbs, scaler_rbs = prepare_data(rbss, rbs_features, rbs_target)

print(f"\nPromoter: X shape {X_prom.shape}, y range [{y_prom.min():.2f}, {y_prom.max():.2f}]")
print(f"RBS: X shape {X_rbs.shape}, y range [{y_rbs.min():.4f}, {y_rbs.max():.4f}]")

# ============================================================
# 4. TRAIN-TEST SPLIT
# ============================================================
X_prom_train, X_prom_test, y_prom_train, y_prom_test = train_test_split(
    X_prom, y_prom, test_size=0.2, random_state=42
)
X_rbs_train, X_rbs_test, y_rbs_train, y_rbs_test = train_test_split(
    X_rbs, y_rbs, test_size=0.2, random_state=42
)

print(f"Promoter - Train: {len(X_prom_train)}, Test: {len(X_prom_test)}")
print(f"RBS - Train: {len(X_rbs_train)}, Test: {len(X_rbs_test)}")

# ============================================================
# 5. TRAIN GPR - REMOVE normalize_y
# ============================================================
def train_gpr(X_train, y_train, name):
    # Better kernel initialization for small samples
    kernel = ConstantKernel(1.0) * RBF(length_scale=1.0) + WhiteKernel(noise_level=0.1)
    gp = GaussianProcessRegressor(
        kernel=kernel,
        n_restarts_optimizer=10,
        alpha=0.01,  # Small regularization
        random_state=42,
        normalize_y=False  # FIXED: Don't normalize y here
    )
    gp.fit(X_train, y_train)
    print(f"\n{name} kernel: {gp.kernel_}")
    return gp

gp_promoter = train_gpr(X_prom_train, y_prom_train, "Promoter")
gp_rbs = train_gpr(X_rbs_train, y_rbs_train, "RBS")

# ============================================================
# 6. EVALUATE
# ============================================================
def evaluate_gpr(gp, X_train, y_train, X_test, y_test, name, is_log=False):
    y_train_pred, _ = gp.predict(X_train, return_std=True)
    y_test_pred, y_test_std = gp.predict(X_test, return_std=True)
    
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    test_mae = mean_absolute_error(y_test, y_test_pred)
    
    print(f"\n{name}:")
    print(f"  Train R2: {train_r2:.4f}")
    print(f"  Test R2: {test_r2:.4f}")
    print(f"  Test RMSE: {test_rmse:.4f}")
    print(f"  Test MAE: {test_mae:.4f}")
    
    return {
        'y_test': y_test,
        'y_pred': y_test_pred,
        'y_std': y_test_std,
        'train_r2': train_r2,
        'test_r2': test_r2,
        'test_rmse': test_rmse,
        'test_mae': test_mae,
        'is_log': is_log
    }

prom_results = evaluate_gpr(gp_promoter, X_prom_train, y_prom_train,
                            X_prom_test, y_prom_test, "Promoter", is_log=False)
rbs_results = evaluate_gpr(gp_rbs, X_rbs_train, y_rbs_train,
                           X_rbs_test, y_rbs_test, "RBS", is_log=True)

# ============================================================
# 7. CROSS-VALIDATION - FIXED
# ============================================================
def cv_score(X, y, name):
    kernel = ConstantKernel(1.0) * RBF(length_scale=1.0) + WhiteKernel(noise_level=0.1)
    gp = GaussianProcessRegressor(kernel=kernel, alpha=0.01, random_state=42, normalize_y=False)
    scores = cross_val_score(gp, X, y, cv=5, scoring='r2')
    print(f"{name} CV: {scores.mean():.4f} +/- {scores.std():.4f}")
    return scores

cv_prom = cv_score(X_prom, y_prom, "Promoter")
cv_rbs = cv_score(X_rbs, y_rbs, "RBS")

# ============================================================
# 8. VISUALIZATION
# ============================================================
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# Promoter Prediction
y_test_prom = prom_results['y_test']
y_pred_prom = prom_results['y_pred']
axes[0, 0].scatter(y_test_prom, y_pred_prom, alpha=0.7, color='steelblue')
min_val, max_val = min(y_test_prom.min(), y_pred_prom.min()), max(y_test_prom.max(), y_pred_prom.max())
axes[0, 0].plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
axes[0, 0].set_xlabel('Actual')
axes[0, 0].set_ylabel('Predicted')
axes[0, 0].set_title(f'Promoter (R2={prom_results["test_r2"]:.3f})')
axes[0, 0].grid(True, alpha=0.3)

# RBS Prediction (Original Scale)
y_test_rbs = 10 ** rbs_results['y_test']
y_pred_rbs = 10 ** rbs_results['y_pred']
axes[0, 1].scatter(y_test_rbs, y_pred_rbs, alpha=0.7, color='coral')
min_val, max_val = min(y_test_rbs.min(), y_pred_rbs.min()), max(y_test_rbs.max(), y_pred_rbs.max())
axes[0, 1].plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
axes[0, 1].set_xlabel('Actual')
axes[0, 1].set_ylabel('Predicted')
axes[0, 1].set_title(f'RBS (R2={rbs_results["test_r2"]:.3f})')
axes[0, 1].grid(True, alpha=0.3)

# Promoter Uncertainty
axes[0, 2].bar(range(len(prom_results['y_std'])), prom_results['y_std'], color='steelblue', alpha=0.7)
axes[0, 2].set_xlabel('Test Sample')
axes[0, 2].set_ylabel('Uncertainty')
axes[0, 2].set_title('Promoter Uncertainty')
axes[0, 2].grid(True, alpha=0.3)

# RBS Uncertainty
axes[1, 0].bar(range(len(rbs_results['y_std'])), rbs_results['y_std'], color='coral', alpha=0.7)
axes[1, 0].set_xlabel('Test Sample')
axes[1, 0].set_ylabel('Uncertainty')
axes[1, 0].set_title('RBS Uncertainty')
axes[1, 0].grid(True, alpha=0.3)

# Promoter Residuals
res_prom = y_test_prom - y_pred_prom
axes[1, 1].scatter(y_pred_prom, res_prom, alpha=0.7, color='steelblue')
axes[1, 1].axhline(y=0, color='r', linestyle='--', linewidth=2)
axes[1, 1].set_xlabel('Predicted')
axes[1, 1].set_ylabel('Residual')
axes[1, 1].set_title('Promoter Residuals')
axes[1, 1].grid(True, alpha=0.3)

# RBS Residuals
res_rbs = y_test_rbs - y_pred_rbs
axes[1, 2].scatter(y_pred_rbs, res_rbs, alpha=0.7, color='coral')
axes[1, 2].axhline(y=0, color='r', linestyle='--', linewidth=2)
axes[1, 2].set_xlabel('Predicted')
axes[1, 2].set_ylabel('Residual')
axes[1, 2].set_title('RBS Residuals')
axes[1, 2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, 'gpr_evaluation_fixed.png'), dpi=150)
print(f"\n[OK] Saved: {os.path.join(PLOT_DIR, 'gpr_evaluation_fixed.png')}")
plt.close()

# ============================================================
# 9. SAVE MODELS AND RESULTS
# ============================================================
joblib.dump(gp_promoter, os.path.join(MODEL_DIR, 'gp_promoter_fixed.pkl'))
joblib.dump(scaler_prom, os.path.join(MODEL_DIR, 'scaler_promoter_fixed.pkl'))
joblib.dump(gp_rbs, os.path.join(MODEL_DIR, 'gp_rbs_fixed.pkl'))
joblib.dump(scaler_rbs, os.path.join(MODEL_DIR, 'scaler_rbs_fixed.pkl'))
print(f"[OK] Models saved to {MODEL_DIR}")

metrics = pd.DataFrame({
    'Model': ['Promoter', 'RBS'],
    'Train_R2': [prom_results['train_r2'], rbs_results['train_r2']],
    'Test_R2': [prom_results['test_r2'], rbs_results['test_r2']],
    'CV_R2_Mean': [cv_prom.mean(), cv_rbs.mean()],
    'CV_R2_Std': [cv_prom.std(), cv_rbs.std()],
    'Test_RMSE': [prom_results['test_rmse'], rbs_results['test_rmse']],
    'Test_MAE': [prom_results['test_mae'], rbs_results['test_mae']]
})
metrics.to_csv(os.path.join(OUTPUT_DIR, 'metrics_fixed.csv'), index=False)
print(f"[OK] Saved: {os.path.join(OUTPUT_DIR, 'metrics_fixed.csv')}")

print("\n" + "=" * 70)
print("GPR TRAINING (FIXED) COMPLETE!")
print("=" * 70)
print(f"""
+-----------------------------------------------------------+
|                    MODEL PERFORMANCE                       |
+-----------------------------------------------------------+
| Model      | Train R2 | Test R2 | CV R2 (5-fold)          |
+-----------------------------------------------------------+
| Promoter   | {prom_results['train_r2']:.4f}   | {prom_results['test_r2']:.4f}   | {cv_prom.mean():.4f} +/- {cv_prom.std():.4f}   |
| RBS (log)  | {rbs_results['train_r2']:.4f}   | {rbs_results['test_r2']:.4f}   | {cv_rbs.mean():.4f} +/- {cv_rbs.std():.4f}   |
+-----------------------------------------------------------+
""")