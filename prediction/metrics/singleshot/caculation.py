import ast
import csv
import re

# 手动指定每个 cluster 的样本数（用于加权）
SAMPLE_COUNTS = {
    "0": 339,
    "1": 18,
    "2": 26,
    "4": 4,
    "5": 12,
}

# 读取 summary_train.txt
with open("summary_train.txt", "r") as f:
    content = f.read()

# 按 cluster 分块
blocks = content.split("cluster ")

metrics = {}  # { cluster_id: {metric_name: last_value} }

def safe_parse_list(s):
    """清理 np.float32(...) / Inf 并转为 Python list"""
    # 注意这里是小写 np
    s = re.sub(r'np\.float32\((.*?)\)', r'\1', s)
    s = s.replace('Inf', "'inf'")
    try:
        values = ast.literal_eval(s)
        values = [float('inf') if v == 'inf' else v for v in values]
        return values
    except Exception as e:
        print(f"⚠️ Unable to parse: {s} ({e})")
        return []

for block in blocks[1:]:  # 第一个是空块，跳过
    lines = block.strip().splitlines()
    if not lines:
        continue

    cluster_id = lines[0].strip(":")  # e.g. "0:"
    cluster_metrics = {}

    for line in lines:
        if ":" in line and "[" in line:
            name, values_str = line.strip().split(": ", 1)
            name = name.strip()  # 例如 "MSE", "RMSE", "MAE", "SSE", "SST"
            values = safe_parse_list(values_str)
            if values:
                cluster_metrics[name] = values[-1]  # 取最后一轮的数值

    metrics[cluster_id] = cluster_metrics
    print(f"Per-Cluster Metrics (last round) -cluster {cluster_id}: {metrics[cluster_id]}")

# ===== 全局加权汇总 =====

total_samples = sum(SAMPLE_COUNTS.values())

# 和日志中的 key 对齐：全大写
weighted_metrics = {
    "MSE": 0.0,
    "RMSE": 0.0,
    "MAE": 0.0,
    "SSE": 0.0,
    "SST": 0.0,
}

for cid, metric in metrics.items():
    count = SAMPLE_COUNTS.get(cid, 0)

    # 跳过没有样本数的 cluster（防御一下）
    if count == 0:
        continue

    # 加权平均：MSE / RMSE / MAE 按样本数加权
    for k in ["MSE", "RMSE", "MAE"]:
        if k in metric:
            weighted_metrics[k] += metric[k] * count / total_samples

    # SSE / SST：这里是 global SSE / SST，直接对各簇求和
    for k in ["SSE", "SST"]:
        if k in metric:
            weighted_metrics[k] += metric[k]

print("\n📊 Final Weighted Global Metrics (last round):")
for k in ["MSE", "RMSE", "MAE", "SSE", "SST"]:
    print(f"{k}: {weighted_metrics[k]:.6f}")

# 用 ΣSSE / ΣSST 计算真正的全局 R²
global_r2 = 1 - (weighted_metrics["SSE"] / weighted_metrics["SST"])
print(f"\n Global R^2 (based on SSE/SST): {global_r2:.6f}")