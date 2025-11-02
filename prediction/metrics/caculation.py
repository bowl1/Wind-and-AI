import ast
import csv

# 手动定义每个 cluster 的样本数量
SAMPLE_COUNTS = {
    "0": 51,
    "2": 4,
    "3": 251,
    "4": 76,
    "5": 6,
    "6": 10,
}

# 读取 txt 文件内容
with open("summary_train.txt", "r") as f:
    content = f.read()

# 分割为每个 cluster 的块
blocks = content.split("cluster ")

metrics = {}  # { cluster_id: {metric_name: last_value} }

for block in blocks[1:]:  # 跳过第一个空块
    lines = block.strip().splitlines()
    cluster_id = lines[0].strip(":")  # 取 ID 字符串
    cluster_metrics = {}
    for line in lines:
        if ":" in line and "[" in line:
            name, values_str = line.strip().split(": ", 1)
            values = ast.literal_eval(values_str)
            cluster_metrics[name.strip()] = values[-1]  # 取最后一轮值
    metrics[cluster_id] = cluster_metrics
    print(f"Per-Cluster Metrics (last round) - cluster {cluster_id}: {metrics[cluster_id]}")

# 汇总指标
total_samples = sum(SAMPLE_COUNTS.values())
weighted_metrics = {
    "MSE": 0,
    "RMSE": 0,
    "MAE": 0,
    "NRMSE": 0,
    "SSE": 0,
    "SST": 0,
}

# 加权累计
for cid, metric in metrics.items():
    count = SAMPLE_COUNTS.get(cid, 0)
    for k in weighted_metrics:
        if k in ["SSE", "SST"]:
            weighted_metrics[k] += metric[k]  # 累加
        else:
            weighted_metrics[k] += metric[k] * count / total_samples  # 样本加权平均

# 输出加权平均结果
print("\n📊 Final Weighted Global Metrics (last round):")
for k, v in weighted_metrics.items():
    print(f"{k}: {v:.6f}")

# 添加标准 R² 计算（基于总 SSE / SST）
global_r2 = 1 - (weighted_metrics["SSE"] / weighted_metrics["SST"])
print(f"\n Global R^2 (based on SSE/SST): {global_r2:.6f}")

