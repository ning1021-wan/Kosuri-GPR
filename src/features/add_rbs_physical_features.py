"""
Add Physical-Chemical Features for RBS
"""

import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 将项目根目录加入路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.config import RAW_DIR, PROCESSED_DIR, FEATURE_DIR, PLOT_DIR_FINAL, PROJECT_ROOT

print("=" * 60)
print("ADDING PHYSICAL FEATURES TO RBS")
print("=" * 60)

# ============================================================
# 1. Load Data
# ============================================================
raw_file = os.path.join(RAW_DIR, 'RBS_data.xlsx')
if not os.path.exists(raw_file):
    print(f"ERROR: {raw_file} not found")
    # 尝试搜索
    print(f"Project root: {PROJECT_ROOT}")
    exit()

rbs_raw = pd.read_excel(raw_file, engine='openpyxl')
print(f"Loaded raw data: {len(rbs_raw)} records")

feat_file = os.path.join(PROCESSED_DIR, 'rbs_features_fixed.csv')
if not os.path.exists(feat_file):
    print(f"ERROR: {feat_file} not found")
    exit()

rbs_feat = pd.read_csv(feat_file)
print(f"Loaded feature data: {len(rbs_feat)} records")

# Merge
rbs_df = rbs_feat.merge(rbs_raw[['Strain', 'Sequence']],
                        left_on='strain', right_on='Strain', how='left')
print(f"Merged: {len(rbs_df)} records")

# ============================================================
# 2. Physical Feature Functions
# ============================================================

def calc_sd_binding_energy(seq, rrna_motif='CCUCCU'):
    seq = str(seq).replace(' ', '').upper().replace('T', 'U')
    best_score = -999
    for i in range(len(seq) - len(rrna_motif) + 1):
        subseq = seq[i:i+len(rrna_motif)]
        score = 0
        for rbs_base, rrna_base in zip(subseq, rrna_motif):
            if (rbs_base == 'G' and rrna_base == 'C') or (rbs_base == 'C' and rrna_base == 'G'):
                score += 2.5
            elif (rbs_base == 'A' and rrna_base == 'U') or (rbs_base == 'U' and rrna_base == 'A'):
                score += 1.5
            elif (rbs_base == 'G' and rrna_base == 'U') or (rbs_base == 'U' and rrna_base == 'G'):
                score += 1.0
            else:
                score -= 1.0
        if score > best_score:
            best_score = score
    return -best_score * 1.2, best_score

def calc_rbs_folding_energy(seq):
    seq = str(seq).replace(' ', '').upper()
    if 'CATATG' in seq:
        start_pos = seq.find('CATATG')
        local_seq = seq[max(0, start_pos-20):start_pos]
    else:
        local_seq = seq[-20:]
    if len(local_seq) < 6:
        return 0
    gc = (local_seq.count('G') + local_seq.count('C')) / len(local_seq)
    stem_score = local_seq.count('GC') + local_seq.count('CG') + local_seq.count('AT') + local_seq.count('TA')
    stem_score = stem_score / len(local_seq)
    return gc * 8 + stem_score * 5

def calc_optimal_spacer_penalty(spacer_length):
    optimal = 7
    if spacer_length == 999:
        return 5
    return abs(spacer_length - optimal) * 1.2

def calc_start_context_score(seq):
    seq = str(seq).replace(' ', '').upper()
    if 'CATATG' not in seq:
        return 0
    start_pos = seq.find('CATATG')
    score = 0
    if start_pos >= 3 and seq[start_pos-3] == 'A':
        score += 1.5
    if start_pos + 7 < len(seq) and seq[start_pos+7] == 'G':
        score += 1.0
    return score

# ============================================================
# 3. Extract Physical Features
# ============================================================
print("\nExtracting physical features...")
physical_features = []

for idx, row in rbs_df.iterrows():
    seq = row['Sequence']
    seq_clean = str(seq).replace(' ', '')
    delta_g, sd_score = calc_sd_binding_energy(seq_clean)
    spacer_len = row.get('spacer_length', 999)
    physical_features.append({
        'sd_delta_g': delta_g,
        'sd_raw_score': sd_score,
        'rbs_folding_energy': calc_rbs_folding_energy(seq_clean),
        'spacer_penalty': calc_optimal_spacer_penalty(spacer_len),
        'start_context_score': calc_start_context_score(seq_clean)
    })

phys_df = pd.DataFrame(physical_features)
rbs_enhanced = pd.concat([rbs_df, phys_df], axis=1)

# ============================================================
# 4. Correlation Analysis
# ============================================================
print("\n" + "-" * 50)
print("CORRELATION ANALYSIS")
print("-" * 50)

target = 'F_on_OD'
exclude_cols = ['strain', 'mean_RNA', 'mean_prot', 'F_on_OD_log', 'Sequence', 'Strain']
feature_cols = [c for c in rbs_enhanced.columns if c not in exclude_cols]

correlations = []
for col in feature_cols:
    if col != target:
        corr = rbs_enhanced[col].corr(rbs_enhanced[target])
        if not np.isnan(corr):
            correlations.append((col, abs(corr)))
correlations.sort(key=lambda x: x[1], reverse=True)

print("\nTop 10 Features by Correlation:")
print("-" * 55)
phys_set = {'sd_delta_g', 'sd_raw_score', 'rbs_folding_energy', 'spacer_penalty', 'start_context_score'}
for i, (feat, corr) in enumerate(correlations[:10], 1):
    tag = " [PHYS]" if feat in phys_set else ""
    print(f"  {i:2d}. {feat:<30} | r = {corr:.4f}{tag}")

# ============================================================
# 5. Save Results
# ============================================================
output_file = os.path.join(PROCESSED_DIR, 'rbs_features_physical.csv')
rbs_enhanced.to_csv(output_file, index=False)
print(f"\nSaved: {output_file}")

# ============================================================
# 6. Visualization
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

top = correlations[:8]
names = [f[0] for f in top]
values = [f[1] for f in top]
colors = ['coral' if n in phys_set else 'steelblue' for n in names]

axes[0].barh(names, values, color=colors, alpha=0.8)
axes[0].set_xlabel('Absolute Correlation')
axes[0].set_title('Top 8 RBS Features (Coral = Physical)')
axes[0].invert_yaxis()

if 'sd_delta_g' in rbs_enhanced.columns:
    axes[1].scatter(rbs_enhanced['sd_delta_g'], rbs_enhanced['F_on_OD'], alpha=0.7, color='coral')
    axes[1].set_xlabel('sd_delta_g (more negative = stronger)')
    axes[1].set_ylabel('RBS Strength (F-on/OD)')
    axes[1].set_title('sd_delta_g vs Strength')

if 'rbs_folding_energy' in rbs_enhanced.columns:
    axes[2].scatter(rbs_enhanced['rbs_folding_energy'], rbs_enhanced['F_on_OD'], alpha=0.7, color='steelblue')
    axes[2].set_xlabel('Folding Energy (higher = more structure)')
    axes[2].set_ylabel('RBS Strength (F-on/OD)')
    axes[2].set_title('Folding Energy vs Strength')

# 使用 config 中的 PLOT_DIR_FINAL
os.makedirs(PLOT_DIR_FINAL, exist_ok=True)
plot_file = os.path.join(PLOT_DIR_FINAL, 'rbs_physical_features.png')
plt.tight_layout()
plt.savefig(plot_file, dpi=150)
print(f"Saved plot: {plot_file}")
plt.show()

print("\n" + "=" * 60)
print("PHYSICAL FEATURES ADDED SUCCESSFULLY!")
print("=" * 60)