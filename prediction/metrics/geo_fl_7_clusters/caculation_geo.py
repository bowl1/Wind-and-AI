import re
import ast
import csv

# === 1. 手动定义每个 cluster 的样本数量 ===
SAMPLE_COUNTS = {
    "0": 73,
    "1": 93,
    "2": 47,
    "3": 36,
    "4": 74,
    "5": 23,
    "6": 54,
}

# === 2. 定义辅助函数：清洗 np.float32(...) 并解析为 Python 列表 ===
def safe_parse_list(s):
    """清理 np.float32(...) 和 inf 表达式，并解析成 Python 列表"""
    # 如果字符串里包含 'inf'，直接跳过解析
    if "inf" in s:
        raise ValueError("includes 'inf', cannot parse safely")

    # 正常清洗 np.float32(...) 语法
    s = re.sub(r'np\.float32\((.*?)\)', r'\1', s)
    return ast.literal_eval(s)

# === 3. 读取 txt 文件内容 ===
with open("summary_train_geo.txt", "r") as f:
    content = f.read()

# === 4. 解析每个 cluster 的指标 ===
blocks = content.split("cluster ")
metrics = {}

for block in blocks[1:]:  # 跳过第一个空块
    lines = block.strip().splitlines()
    cluster_id = lines[0].strip(":")  # 取 cluster ID
    cluster_metrics = {}
    for line in lines:
        if ":" in line and "[" in line:
            name, values_str = line.strip().split(": ", 1)
            try:
                values = safe_parse_list(values_str)
                cluster_metrics[name.strip()] = float(values[-1])  # 取最后一轮的值
            except Exception as e:
                print(f"⚠️ the reason for skiping lines ({name}): {values_str}\nreason: {e}")
    metrics[cluster_id] = cluster_metrics
    print(f"Cluster {cluster_id} metrics (last round): {cluster_metrics}")

# === 5. 计算加权平均指标 ===
total_samples = sum(SAMPLE_COUNTS.values())
weighted_metrics = {k: 0 for k in ["MSE", "RMSE", "MAE", "NRMSE", "SSE", "SST"]}

for cid, metric in metrics.items():
    count = SAMPLE_COUNTS.get(cid, 0)
    for k in weighted_metrics:
        if k in ["SSE", "SST"]:
            weighted_metrics[k] += metric.get(k, 0.0)
        else:
            weighted_metrics[k] += metric.get(k, 0.0) * count / total_samples

# === 6. 输出汇总指标 ===
print("\n📊 Final Weighted Global Metrics (last round):")
for k, v in weighted_metrics.items():
    print(f"{k}: {v:.6f}")

# === 7. 计算标准 R² ===
global_r2 = 1 - (weighted_metrics["SSE"] / weighted_metrics["SST"])
print(f"\nGlobal R^2 (based on total SSE/SST): {global_r2:.6f}")

# === 8. 保存为 CSV 文件 ===
with open("geo_test_summary.csv", "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["Metric", "WeightedValue"])
    for k, v in weighted_metrics.items():
        writer.writerow([k, v])
    writer.writerow(["Global_R2", global_r2])

print("\n Saved results to geo_test_summary.csv")