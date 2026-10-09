"""
Active Learning Recommendation
使用训练好的GPR模型推荐最有价值的湿实验候选
"""

import sys
import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import joblib
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# 路径配置（直接定义，不依赖 config）
# ============================================================
# 获取当前脚本所在目录
script_dir = os.path.dirname(os.path.abspath(__file__))
# 项目根目录（feature_engineering）
project_root = os.path.dirname(script_dir)

# 数据目录
PROCESSED_DATA_DIR = os.path.join(project_root, 'data', 'processed')
# 模型目录
MODEL_DIR = os.path.join(project_root, 'results', 'gp_results', 'models')
# 输出目录
OUTPUT_DIR = os.path.join(project_root, 'results', 'active_learning')

# 确保输出目录存在
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 70)
print("ACTIVE LEARNING RECOMMENDATION")
print("=" * 70)
print(f"Project root: {project_root}")
print(f"Data dir: {PROCESSED_DATA_DIR}")
print(f"Model dir: {MODEL_DIR}")
print(f"Output dir: {OUTPUT_DIR}")

# ============================================================
# 1. 加载数据和模型
# ============================================================
# 加载特征数据
promoters = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, 'promoter_features_optimized.csv'))
rbss = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, 'rbs_features_optimized.csv'))

print(f"Promoters: {len(promoters)}")
print(f"RBS: {len(rbss)}")

# 加载训练好的模型 (使用fix版本)
gp_promoter = joblib.load(os.path.join(MODEL_DIR, 'gp_promoter_fixed.pkl'))
scaler_promoter = joblib.load(os.path.join(MODEL_DIR, 'scaler_promoter_fixed.pkl'))
gp_rbs = joblib.load(os.path.join(MODEL_DIR, 'gp_rbs_fixed.pkl'))
scaler_rbs = joblib.load(os.path.join(MODEL_DIR, 'scaler_rbs_fixed.pkl'))

print("\n[OK] Models loaded")

# ============================================================
# 2. 定义特征列（必须与训练时一致）
# ============================================================
promoter_features = ['gc_motif_density', 'score_minus10']
rbs_features = ['sd_best_score', 'spacer_gc']

# ============================================================
# 3. 生成所有候选组合（启动子 × RBS）
# ============================================================
print("\n" + "-" * 50)
print("GENERATING CANDIDATE COMBINATIONS")
print("-" * 50)

# 提取每个启动子的特征向量
promoter_vectors = promoters[promoter_features].values
promoter_strains = promoters['Strain'].values
promoter_sequences = promoters['Sequence'].values

# 提取每个RBS的特征向量
rbs_vectors = rbss[rbs_features].values
rbs_strains = rbss['Strain'].values
rbs_sequences = rbss['Sequence'].values

# 生成所有组合
candidates = []
for i, (p_vec, p_strain, p_seq) in enumerate(zip(promoter_vectors, promoter_strains, promoter_sequences)):
    for j, (r_vec, r_strain, r_seq) in enumerate(zip(rbs_vectors, rbs_strains, rbs_sequences)):
        candidates.append({
            'promoter_strain': int(p_strain),
            'promoter_seq': p_seq[:50] + '...' if len(str(p_seq)) > 50 else p_seq,
            'promoter_features': p_vec.tolist(),
            'rbs_strain': int(r_strain),
            'rbs_seq': r_seq[:50] + '...' if len(str(r_seq)) > 50 else r_seq,
            'rbs_features': r_vec.tolist(),
            'combination_id': f"P{p_strain}_R{r_strain}"
        })

print(f"Total combinations: {len(candidates)}")

# ============================================================
# 4. 构建特征矩阵并预测
# ============================================================
print("\n" + "-" * 50)
print("PREDICTING WITH GPR MODELS")
print("-" * 50)

# 构建Promoter特征矩阵并标准化
X_prom_candidates = np.array([c['promoter_features'] for c in candidates])
X_prom_scaled = scaler_promoter.transform(X_prom_candidates)

# Promoter预测
y_prom_pred, y_prom_std = gp_promoter.predict(X_prom_scaled, return_std=True)

# 构建RBS特征矩阵并标准化
X_rbs_candidates = np.array([c['rbs_features'] for c in candidates])
X_rbs_scaled = scaler_rbs.transform(X_rbs_candidates)

# RBS预测 (log scale)
y_rbs_pred_log, y_rbs_std = gp_rbs.predict(X_rbs_scaled, return_std=True)

# 转换为原始尺度
y_rbs_pred = 10 ** y_rbs_pred_log

# 保存预测结果到候选列表
for i, c in enumerate(candidates):
    c['promoter_pred'] = float(y_prom_pred[i])
    c['promoter_uncertainty'] = float(y_prom_std[i])
    c['rbs_pred'] = float(y_rbs_pred[i])
    c['rbs_uncertainty'] = float(y_rbs_std[i])

# ============================================================
# 5. 主动学习推荐策略
# ============================================================
print("\n" + "-" * 50)
print("RECOMMENDATION STRATEGY")
print("-" * 50)

print("""
推荐策略:
  1. 优先选择预测不确定性最高的组合 (探索性)
  2. 兼顾预测强度 (利用现有知识)
  3. 覆盖不同的启动子和RBS，分散探索

推荐类别:
  - 'High Uncertainty': 模型最不确定，探索价值最高
  - 'High Strength': 预测强度最高，验证潜力最优
  - 'Balanced': 不确定性和预测强度均衡
""")

# ============================================================
# 6. Top 推荐列表
# ============================================================
print("\n" + "-" * 50)
print("TOP RECOMMENDATIONS")
print("-" * 50)

# --- 6a. 最高不确定性 (Promoter) ---
top_prom_uncertainty = sorted(candidates, key=lambda x: x['promoter_uncertainty'], reverse=True)[:10]

# --- 6b. 最高不确定性 (RBS) ---
top_rbs_uncertainty = sorted(candidates, key=lambda x: x['rbs_uncertainty'], reverse=True)[:10]

# --- 6c. 最高预测强度 (Promoter) ---
top_prom_strength = sorted(candidates, key=lambda x: x['promoter_pred'], reverse=True)[:10]

# --- 6d. 最高预测强度 (RBS) ---
top_rbs_strength = sorted(candidates, key=lambda x: x['rbs_pred'], reverse=True)[:10]

# --- 6e. 均衡推荐 (不确定性+强度兼顾) ---
def balance_score(c):
    # 归一化后的不确定性 + 归一化后的强度
    norm_unc = c['promoter_uncertainty'] / (max([c2['promoter_uncertainty'] for c2 in candidates]) + 1e-9)
    norm_strength = c['promoter_pred'] / (max([c2['promoter_pred'] for c2 in candidates]) + 1e-9)
    return norm_unc * 0.7 + norm_strength * 0.3  # 偏向不确定性

top_balanced = sorted(candidates, key=balance_score, reverse=True)[:10]

# ============================================================
# 7. 输出推荐结果
# ============================================================
def print_recommendations(rec_list, title, n=10):
    print(f"\n{title}")
    print("-" * 70)
    print(f"{'Rank':<6} {'Combination':<18} {'Promoter':<15} {'RBS':<15} {'P_Strength':<12} {'P_Uncertainty':<12}")
    print("-" * 70)
    for i, c in enumerate(rec_list[:n], 1):
        print(f"{i:<6} {c['combination_id']:<18} {c['promoter_seq'][:12]:<15} {c['rbs_seq'][:12]:<15} {c['promoter_pred']:<12.0f} {c['promoter_uncertainty']:<12.2f}")

# 打印各类推荐
print_recommendations(top_prom_uncertainty, "PROMOTER: HIGHEST UNCERTAINTY", 10)
print_recommendations(top_prom_strength, "PROMOTER: HIGHEST PREDICTED STRENGTH", 10)
print_recommendations(top_balanced, "PROMOTER: BALANCED (Uncertainty + Strength)", 10)

# 同样为RBS输出简要推荐
print("\n\n" + "-" * 50)
print("RBS RECOMMENDATIONS")
print("-" * 50)

print_recommendations(top_rbs_uncertainty, "RBS: HIGHEST UNCERTAINTY", 10)
print_recommendations(top_rbs_strength, "RBS: HIGHEST PREDICTED STRENGTH", 10)

# ============================================================
# 8. 保存结果到CSV
# ============================================================
print("\n" + "-" * 50)
print("SAVING RESULTS")
print("-" * 50)

# 完整的候选列表
all_candidates_df = pd.DataFrame(candidates)
all_candidates_df.to_csv(os.path.join(OUTPUT_DIR, 'all_candidates.csv'), index=False)

# Top推荐汇总
recommendations = {
    'promoter_high_uncertainty': top_prom_uncertainty[:10],
    'promoter_high_strength': top_prom_strength[:10],
    'promoter_balanced': top_balanced[:10],
    'rbs_high_uncertainty': top_rbs_uncertainty[:10],
    'rbs_high_strength': top_rbs_strength[:10],
}

# 保存每个推荐列表到CSV
for key, rec_list in recommendations.items():
    df = pd.DataFrame(rec_list)
    df.to_csv(os.path.join(OUTPUT_DIR, f'{key}.csv'), index=False)

print(f"[OK] Saved all candidates to: {OUTPUT_DIR}/all_candidates.csv")
print(f"[OK] Saved recommendations to: {OUTPUT_DIR}/")

# ============================================================
# 9. 生成推荐汇总报告
# ============================================================
print("\n" + "-" * 50)
print("RECOMMENDATION SUMMARY")
print("-" * 50)

# 提取Top 5推荐
top_5 = top_balanced[:5]
print("\nTop 5 Recommended Combinations (Balanced):")
print("=" * 70)
for i, c in enumerate(top_5, 1):
    print(f"{i}. {c['combination_id']}")
    print(f"   Promoter: Strain {c['promoter_strain']}")
    print(f"   RBS:      Strain {c['rbs_strain']}")
    print(f"   Predicted Strength: {c['promoter_pred']:.0f} (uncertainty: {c['promoter_uncertainty']:.2f})")
    print(f"   RBS Predicted: {c['rbs_pred']:.0f} (uncertainty: {c['rbs_uncertainty']:.2f})")
    print()

# ============================================================
# 10. Active Learning Report
# ============================================================
report = f"""
======================================================================
ACTIVE LEARNING RECOMMENDATION REPORT
======================================================================

Total candidates evaluated: {len(candidates)}
Number of Promoters: {len(promoters)}
Number of RBS: {len(rbss)}

RECOMMENDATION STRATEGY:
- Prioritize high uncertainty for exploration (70% weight)
- Balance with predicted strength (30% weight)
- Total recommended: 10 combinations

TOP 5 RECOMMENDED COMBINATIONS:
"""

for i, c in enumerate(top_5[:5], 1):
    report += f"""
{i}. {c['combination_id']}
   - Promoter Strain: {c['promoter_strain']}
   - RBS Strain: {c['rbs_strain']}
   - Promoter Strength Prediction: {c['promoter_pred']:.0f} +/- {c['promoter_uncertainty']:.2f}
   - RBS Strength Prediction: {c['rbs_pred']:.0f} +/- {c['rbs_uncertainty']:.2f}
"""

report += """
======================================================================
NEXT STEPS:
1. Perform wet lab experiments on the 5-10 recommended combinations
2. Measure actual expression strength
3. Feed back experimental data to update the GPR model
4. Run a second round of active learning
======================================================================
"""

with open(os.path.join(OUTPUT_DIR, 'active_learning_report.txt'), 'w') as f:
    f.write(report)

print(f"[OK] Saved report to: {OUTPUT_DIR}/active_learning_report.txt")

print("\n" + "=" * 70)
print("ACTIVE LEARNING RECOMMENDATION COMPLETE!")
print("=" * 70)
print(f"""
Output files:
  - {OUTPUT_DIR}/all_candidates.csv
  - {OUTPUT_DIR}/promoter_high_uncertainty.csv
  - {OUTPUT_DIR}/promoter_high_strength.csv
  - {OUTPUT_DIR}/promoter_balanced.csv
  - {OUTPUT_DIR}/rbs_high_uncertainty.csv
  - {OUTPUT_DIR}/rbs_high_strength.csv
  - {OUTPUT_DIR}/active_learning_report.txt

Next: Perform wet lab experiments on recommended combinations,
      then update model with new data.
""")