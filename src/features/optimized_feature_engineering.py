"""
Optimized Feature Engineering for Promoter and RBS
Task 3: Core Feature Extraction (Without Physical Features)
"""

import sys
import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 非交互式后端

import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# 路径配置（直接定义，不依赖 config 中的 PLOT_DIR）
# ============================================================
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

# 从 config 导入基本路径
from src.utils.config import (
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    FEATURE_RESULTS_DIR
)

# 明确设置绘图目录
PLOT_DIR = os.path.join(FEATURE_RESULTS_DIR, 'plots')

# 确保所有输出目录存在
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
os.makedirs(FEATURE_RESULTS_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)

print("=" * 70)
print("OPTIMIZED FEATURE ENGINEERING")
print("=" * 70)
print(f"Project root: {project_root}")
print(f"Raw data dir: {RAW_DATA_DIR}")
print(f"Output dir: {PROCESSED_DATA_DIR}")
print(f"Plot dir: {PLOT_DIR}")

# ============================================================
# 1. 加载数据
# ============================================================
print("\n" + "-" * 50)
print("LOADING DATA")
print("-" * 50)

promoter_file = os.path.join(RAW_DATA_DIR, 'promoter_data.xlsx')
rbs_file = os.path.join(RAW_DATA_DIR, 'RBS_data.xlsx')

if not os.path.exists(promoter_file):
    print(f"ERROR: {promoter_file} not found")
    print(f"Available files: {os.listdir(RAW_DATA_DIR)}")
    sys.exit(1)

if not os.path.exists(rbs_file):
    print(f"ERROR: {rbs_file} not found")
    sys.exit(1)

promoters_raw = pd.read_excel(promoter_file, engine='openpyxl')
rbss_raw = pd.read_excel(rbs_file, engine='openpyxl')

print(f"Promoters: {len(promoters_raw)} samples")
print(f"RBS: {len(rbss_raw)} samples")

# ============================================================
# 2. 通用特征函数
# ============================================================
print("\n" + "-" * 50)
print("DEFINING FEATURE FUNCTIONS")
print("-" * 50)

def gc_content(seq):
    seq = str(seq).replace(' ', '')
    if len(seq) == 0:
        return 0
    return (seq.count('G') + seq.count('C')) / len(seq)

def at_content(seq):
    seq = str(seq).replace(' ', '')
    if len(seq) == 0:
        return 0
    return (seq.count('A') + seq.count('T')) / len(seq)

def gc_skew(seq):
    seq = str(seq).replace(' ', '')
    g = seq.count('G')
    c = seq.count('C')
    if g + c == 0:
        return 0
    return (g - c) / (g + c)

def hamming_distance(seq, consensus):
    seq = str(seq).replace(' ', '')
    min_len = min(len(seq), len(consensus))
    return sum(1 for i in range(min_len) if seq[i] != consensus[i])

def count_motif(seq, motif):
    seq = str(seq).replace(' ', '')
    return seq.count(motif)

def sequence_complexity(seq):
    seq = str(seq).replace(' ', '')
    if len(seq) < 3:
        return 0
    kmers = set(seq[i:i+3] for i in range(len(seq)-2))
    return len(kmers) / (len(seq) - 2)

def approximate_mfe(seq):
    gc = gc_content(seq)
    return -gc * 15

# ============================================================
# 3. 启动子特征
# ============================================================
print("\n" + "-" * 50)
print("EXTRACTING PROMOTER FEATURES")
print("-" * 50)

def extract_promoter_features(df):
    df = df.copy()
    df['seq_clean'] = df['Sequence'].str.replace(' ', '')
    
    df['gc_content'] = df['seq_clean'].apply(gc_content)
    df['at_content'] = df['seq_clean'].apply(at_content)
    df['gc_skew'] = df['seq_clean'].apply(gc_skew)
    df['length'] = df['seq_clean'].apply(len)
    df['complexity'] = df['seq_clean'].apply(sequence_complexity)
    df['mfe'] = df['seq_clean'].apply(approximate_mfe)
    
    df['gc_motif_count'] = df['seq_clean'].apply(lambda s: count_motif(s, 'GC') + count_motif(s, 'CG'))
    df['gc_motif_density'] = df['gc_motif_count'] / df['length']
    df['at_motif_count'] = df['seq_clean'].apply(lambda s: count_motif(s, 'AT') + count_motif(s, 'TA'))
    df['at_motif_density'] = df['at_motif_count'] / df['length']
    
    df['core_seq'] = df['seq_clean'].apply(lambda s: s[8:] if len(s) > 8 else s)
    
    df['dist_minus35'] = df['core_seq'].apply(
        lambda s: hamming_distance(s[:6], 'TTGACA') if len(s) >= 6 else 999
    )
    df['score_minus35'] = 6 - df['dist_minus35']
    
    df['dist_minus10'] = df['core_seq'].apply(
        lambda s: hamming_distance(s[12:18], 'TATAAT') if len(s) >= 18 else 999
    )
    df['score_minus10'] = 6 - df['dist_minus10']
    
    df['up_at_content'] = df['core_seq'].apply(
        lambda s: at_content(s[:20]) if len(s) >= 20 else 0
    )
    
    df['spacer_length'] = df['core_seq'].apply(
        lambda s: len(s[6:12]) if len(s) >= 12 else 0
    )
    df['spacer_gc'] = df['core_seq'].apply(
        lambda s: gc_content(s[6:12]) if len(s) >= 12 else 0
    )
    
    df['has_extended_minus10'] = df['core_seq'].apply(
        lambda s: 1 if 'TG' in s[10:16] else 0
    )
    
    df['discriminator_gc'] = df['core_seq'].apply(
        lambda s: gc_content(s[-6:]) if len(s) >= 6 else 0
    )
    
    df['F_on_OD'] = df['F-on/OD']
    df['F_on_OD_log'] = np.log10(df['F-on/OD'] + 1)
    
    return df

promoters = extract_promoter_features(promoters_raw)

print(f"Promoter features: {len(promoters)} rows, {len(promoters.columns)} columns")
print("\nFirst 10 promoters (selected features):")
selected_cols = ['Strain', 'gc_content', 'gc_motif_density', 'score_minus10', 
                 'up_at_content', 'F_on_OD']
print(promoters[selected_cols].head(10).to_string(index=False))

# ============================================================
# 4. RBS特征
# ============================================================
print("\n" + "-" * 50)
print("EXTRACTING RBS FEATURES (WITH SD MISMATCH DETECTION)")
print("-" * 50)

def find_sd_with_mismatch(seq, motif='AGGAGG', max_mismatch=2):
    seq = str(seq).replace(' ', '')
    motif_len = len(motif)
    for i in range(len(seq) - motif_len + 1):
        mismatches = sum(1 for a, b in zip(seq[i:i+motif_len], motif) if a != b)
        if mismatches <= max_mismatch:
            return i, mismatches
    return None, None

def calculate_sd_best_score(seq, motif='AGGAGG'):
    seq = str(seq).replace(' ', '')
    motif_len = len(motif)
    best_score = 0
    for i in range(len(seq) - motif_len + 1):
        matches = sum(1 for a, b in zip(seq[i:i+motif_len], motif) if a == b)
        score = matches / motif_len
        if score > best_score:
            best_score = score
    return best_score

def extract_rbs_features(df):
    df = df.copy()
    df['seq_clean'] = df['Sequence'].str.replace(' ', '')
    
    df['gc_content'] = df['seq_clean'].apply(gc_content)
    df['at_content'] = df['seq_clean'].apply(at_content)
    df['gc_skew'] = df['seq_clean'].apply(gc_skew)
    df['length'] = df['seq_clean'].apply(len)
    df['complexity'] = df['seq_clean'].apply(sequence_complexity)
    df['mfe'] = df['seq_clean'].apply(approximate_mfe)
    
    df['rbs_seq'] = df['seq_clean'].apply(
        lambda s: s[:-6] if s.endswith('CATATG') else s
    )
    
    sd_data = []
    for seq in df['rbs_seq']:
        pos, mismatches = find_sd_with_mismatch(seq, 'AGGAGG', max_mismatch=2)
        sd_data.append({
            'sd_present': 1 if pos is not None else 0,
            'sd_mismatches': mismatches if pos is not None else 999,
            'sd_start_pos': pos if pos is not None else 999,
            'sd_end_pos': pos + 6 if pos is not None else 999,
            'sd_sequence_score': 6 - mismatches if pos is not None else 0
        })
    sd_df = pd.DataFrame(sd_data)
    df = pd.concat([df, sd_df], axis=1)
    
    df['sd_best_score'] = df['rbs_seq'].apply(calculate_sd_best_score)
    
    def calc_sd_distance(row):
        if row['sd_present'] == 1:
            start_codon_pos = len(row['rbs_seq'])
            return start_codon_pos - row['sd_end_pos']
        return 999
    df['sd_to_start_distance'] = df.apply(calc_sd_distance, axis=1)
    
    def calc_spacer(row):
        if row['sd_present'] == 1:
            spacer = row['rbs_seq'][row['sd_end_pos']:len(row['rbs_seq'])]
            return len(spacer)
        return 999
    def calc_spacer_gc(row):
        if row['sd_present'] == 1:
            spacer = row['rbs_seq'][row['sd_end_pos']:len(row['rbs_seq'])]
            return gc_content(spacer) if len(spacer) > 0 else 0
        return 0
    def calc_spacer_at(row):
        if row['sd_present'] == 1:
            spacer = row['rbs_seq'][row['sd_end_pos']:len(row['rbs_seq'])]
            return at_content(spacer) if len(spacer) > 0 else 0
        return 0
    
    df['spacer_length'] = df.apply(calc_spacer, axis=1)
    df['spacer_gc'] = df.apply(calc_spacer_gc, axis=1)
    df['spacer_at'] = df.apply(calc_spacer_at, axis=1)
    
    df['a_rich_before_start'] = df['rbs_seq'].apply(
        lambda s: at_content(s[-15:]) if len(s) >= 15 else at_content(s)
    )
    
    df['F_on_OD'] = df['F-on/OD']
    df['F_on_OD_log'] = np.log10(df['F-on/OD'] + 1)
    
    return df

rbss = extract_rbs_features(rbss_raw)

print(f"RBS features: {len(rbss)} rows, {len(rbss.columns)} columns")
print(f"\nSD motif found: {(rbss['sd_present'] == 1).sum()} / {len(rbss)} ({(rbss['sd_present'] == 1).sum() / len(rbss) * 100:.1f}%)")

print("\nFirst 10 RBS (selected features):")
selected_cols = ['Strain', 'sd_present', 'sd_mismatches', 'sd_best_score', 
                 'spacer_length', 'F_on_OD']
print(rbss[selected_cols].head(10).to_string(index=False))

# ============================================================
# 5. 保存特征数据
# ============================================================
print("\n" + "-" * 50)
print("SAVING FEATURE DATA")
print("-" * 50)

promoter_output = os.path.join(PROCESSED_DATA_DIR, 'promoter_features_optimized.csv')
rbs_output = os.path.join(PROCESSED_DATA_DIR, 'rbs_features_optimized.csv')

promoters.to_csv(promoter_output, index=False)
rbss.to_csv(rbs_output, index=False)

print(f"[OK] Saved: {promoter_output}")
print(f"[OK] Saved: {rbs_output}")

# ============================================================
# 6. 相关性分析
# ============================================================
print("\n" + "-" * 50)
print("CORRELATION ANALYSIS")
print("-" * 50)

def analyze_correlations(df, name, target='F_on_OD', exclude_cols=[]):
    numeric_cols = []
    for col in df.columns:
        if col == target or col in exclude_cols:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            numeric_cols.append(col)
    
    correlations = []
    for col in numeric_cols:
        corr = df[col].corr(df[target])
        if not np.isnan(corr):
            correlations.append((col, abs(corr)))
    
    correlations.sort(key=lambda x: x[1], reverse=True)
    
    print(f"\n{name} - Top 10 Features by Correlation:")
    print("-" * 55)
    for i, (feat, corr) in enumerate(correlations[:10], 1):
        print(f"  {i:2d}. {feat:<30} | r = {corr:.4f}")
    
    return correlations

prom_corr = analyze_correlations(
    promoters, "Promoter",
    exclude_cols=['F_on_OD', 'F_on_OD_log', 'Strain', 'Sequence', 'seq_clean', 'core_seq']
)

rbs_corr = analyze_correlations(
    rbss, "RBS",
    exclude_cols=['F_on_OD', 'F_on_OD_log', 'Strain', 'Sequence', 'seq_clean', 'rbs_seq']
)

# 保存相关性结果
prom_corr_df = pd.DataFrame(prom_corr, columns=['feature', 'abs_correlation'])
prom_corr_df.to_csv(os.path.join(PROCESSED_DATA_DIR, 'promoter_correlations.csv'), index=False)

rbs_corr_df = pd.DataFrame(rbs_corr, columns=['feature', 'abs_correlation'])
rbs_corr_df.to_csv(os.path.join(PROCESSED_DATA_DIR, 'rbs_correlations.csv'), index=False)

print(f"\n[OK] Correlation results saved to {PROCESSED_DATA_DIR}/")

# ============================================================
# 7. 可视化
# ============================================================
print("\n" + "-" * 50)
print("GENERATING VISUALIZATIONS")
print("-" * 50)

# 确保绘图目录存在
os.makedirs(PLOT_DIR, exist_ok=True)

# Figure 1: Top 8 features bar chart
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

top_prom = prom_corr[:8]
names_prom = [f[0] for f in top_prom]
vals_prom = [f[1] for f in top_prom]
axes[0].barh(names_prom, vals_prom, color='steelblue', alpha=0.8)
axes[0].set_xlabel('Absolute Correlation')
axes[0].set_title('Promoter: Top 8 Features')
axes[0].invert_yaxis()

top_rbs = rbs_corr[:8]
names_rbs = [f[0] for f in top_rbs]
vals_rbs = [f[1] for f in top_rbs]
axes[1].barh(names_rbs, vals_rbs, color='coral', alpha=0.8)
axes[1].set_xlabel('Absolute Correlation')
axes[1].set_title('RBS: Top 8 Features')
axes[1].invert_yaxis()

plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, 'feature_correlation_top8.png'), dpi=150)
print(f"[OK] Saved: {os.path.join(PLOT_DIR, 'feature_correlation_top8.png')}")
plt.close()

# Figure 2: Best feature vs strength scatter
fig2, axes2 = plt.subplots(1, 2, figsize=(12, 5))

if len(prom_corr) > 0:
    best_prom = prom_corr[0][0]
    axes2[0].scatter(promoters[best_prom], promoters['F_on_OD'], alpha=0.7, color='steelblue')
    axes2[0].set_xlabel(best_prom)
    axes2[0].set_ylabel('Promoter Strength (F-on/OD)')
    axes2[0].set_title(f'Promoter: {best_prom} vs Strength')

if len(rbs_corr) > 0:
    best_rbs = rbs_corr[0][0]
    axes2[1].scatter(rbss[best_rbs], rbss['F_on_OD'], alpha=0.7, color='coral')
    axes2[1].set_xlabel(best_rbs)
    axes2[1].set_ylabel('RBS Strength (F-on/OD)')
    axes2[1].set_title(f'RBS: {best_rbs} vs Strength')

plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, 'top_feature_vs_strength.png'), dpi=150)
print(f"[OK] Saved: {os.path.join(PLOT_DIR, 'top_feature_vs_strength.png')}")
plt.close()

# Figure 3: Strength distribution
fig3, axes3 = plt.subplots(1, 2, figsize=(12, 4))

axes3[0].hist(promoters['F_on_OD'], bins=15, color='steelblue', edgecolor='white', alpha=0.8)
axes3[0].set_xlabel('Promoter Strength (F-on/OD)')
axes3[0].set_ylabel('Frequency')
axes3[0].set_title('Promoter Strength Distribution')

axes3[1].hist(rbss['F_on_OD'], bins=15, color='coral', edgecolor='white', alpha=0.8)
axes3[1].set_xlabel('RBS Strength (F-on/OD)')
axes3[1].set_ylabel('Frequency')
axes3[1].set_title('RBS Strength Distribution')

plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, 'strength_distribution.png'), dpi=150)
print(f"[OK] Saved: {os.path.join(PLOT_DIR, 'strength_distribution.png')}")
plt.close()

# ============================================================
# 8. 特征摘要
# ============================================================
print("\n" + "-" * 50)
print("FEATURE SUMMARY")
print("-" * 50)

print(f"""
+-----------------------------------------------------------+
|                    FEATURE SUMMARY                         |
+-----------------------------------------------------------+
| Dataset           | Samples | Features | Target           |
+-----------------------------------------------------------+
| Promoter          | {len(promoters)}    | {len(promoters.columns)-6}     | F_on_OD ({promoters['F_on_OD'].min():.1f}-{promoters['F_on_OD'].max():.1f}) |
| RBS               | {len(rbss)}    | {len(rbss.columns)-6}     | F_on_OD ({rbss['F_on_OD'].min():.1f}-{rbss['F_on_OD'].max():.1f}) |
+-----------------------------------------------------------+
""")

print(f"Top feature for Promoter: {prom_corr[0][0]} (r={prom_corr[0][1]:.4f})")
print(f"Top feature for RBS: {rbs_corr[0][0]} (r={rbs_corr[0][1]:.4f})")

# ============================================================
# 9. 推荐特征
# ============================================================
print("\n" + "-" * 50)
print("RECOMMENDED FEATURES FOR GPR")
print("-" * 50)

promoter_recommended = [f[0] for f in prom_corr[:6]]
rbs_recommended = [f[0] for f in rbs_corr[:6]]

print("\nPromoter (Top 6):")
for i, f in enumerate(promoter_recommended, 1):
    corr = prom_corr[i-1][1]
    print(f"  {i}. {f} (r = {corr:.4f})")

print("\nRBS (Top 6):")
for i, f in enumerate(rbs_recommended, 1):
    corr = rbs_corr[i-1][1]
    print(f"  {i}. {f} (r = {corr:.4f})")

# 保存推荐特征到文件
rec_file = os.path.join(FEATURE_RESULTS_DIR, 'recommended_features.txt')
with open(rec_file, 'w') as f:
    f.write("=" * 70 + "\n")
    f.write("RECOMMENDED FEATURES FOR GAUSSIAN PROCESS REGRESSION\n")
    f.write("=" * 70 + "\n\n")
    f.write("PROMOTER FEATURES (Top 6 by correlation):\n")
    for i, feat in enumerate(promoter_recommended, 1):
        corr = prom_corr[i-1][1]
        f.write(f"  {i}. {feat} (r = {corr:.4f})\n")
    f.write("\nRBS FEATURES (Top 6 by correlation):\n")
    for i, feat in enumerate(rbs_recommended, 1):
        corr = rbs_corr[i-1][1]
        f.write(f"  {i}. {feat} (r = {corr:.4f})\n")
    f.write("\n" + "=" * 70 + "\n")
    f.write(f"Promoter target: F_on_OD\n")
    f.write(f"RBS target: F_on_OD_log (recommended for better performance)\n")
    f.write("=" * 70 + "\n")

print(f"\n[OK] Saved recommendations to: {rec_file}")

# ============================================================
# 10. Complete
# ============================================================
print("\n" + "=" * 70)
print("OPTIMIZED FEATURE ENGINEERING COMPLETE!")
print("=" * 70)
print(f"""
Output files:
  - {PROCESSED_DATA_DIR}/promoter_features_optimized.csv
  - {PROCESSED_DATA_DIR}/rbs_features_optimized.csv
  - {PROCESSED_DATA_DIR}/promoter_correlations.csv
  - {PROCESSED_DATA_DIR}/rbs_correlations.csv
  - {PLOT_DIR}/*.png
  - {rec_file}

Next step: Train GPR models using src/models/train_gpr.py
""")