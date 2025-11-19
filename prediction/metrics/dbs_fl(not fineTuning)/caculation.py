import ast
import csv
import re

# Manually define the number of samples per cluster
SAMPLE_COUNTS = {
    "0": 51,
    "2": 4,
    "3": 251,
    "4": 76,
    "5": 6,
    "6": 10,
}

# Read the contents of txt file
with open("summary_test.txt", "r") as f:
    content = f.read()

# Split into chunks for each cluster
blocks = content.split("cluster ")

metrics = {}  # { cluster_id: {metric_name: last_value} }

def safe_parse_list(s):
    """清理 np.float32(...)、inf，并转换为 Python 列表"""
    s = re.sub(r'np\.float32\((.*?)\)', r'\1', s)
    s = s.replace('inf', "'inf'")
    try:
        values = ast.literal_eval(s)
        values = [float('inf') if v == 'inf' else v for v in values]
        return values
    except Exception as e:
        print(f"⚠️ 无法解析: {s} ({e})")
        return []

for block in blocks[1:]:  # skip first empty block
    lines = block.strip().splitlines()
    cluster_id = lines[0].strip(":")  # Get ID string
    cluster_metrics = {}
    for line in lines:
        if ":" in line and "[" in line:
            name, values_str = line.strip().split(": ", 1)
            try:
                values = safe_parse_list(values_str)
                if values:
                    cluster_metrics[name.strip()] = values[-1]  # Get the last round value
            except Exception as e:
                print(f"⚠️ 跳过无法解析的行 ({name}): {values_str}\n原因: {e}")
    metrics[cluster_id] = cluster_metrics
    print(f"Per-Cluster Metrics (last round) - cluster {cluster_id}: {metrics[cluster_id]}")

# Summary indicators
total_samples = sum(SAMPLE_COUNTS.values())
weighted_metrics = {
    "MSE": 0,
    "RMSE": 0,
    "MAE": 0,
    # "NRMSE": 0,
    "SSE": 0,
    "SST": 0,
}

# weighted accumulation
for cid, metric in metrics.items():
    count = SAMPLE_COUNTS.get(cid, 0)
    for k in weighted_metrics:
        if k in ["SSE", "SST"]:
            weighted_metrics[k] += metric[k]  # Accumulate
        else:
            weighted_metrics[k] += metric[k] * count / total_samples  # sample weighted average

# Output weighted average results
print("\n📊 Final Weighted Global Metrics (last round):")
for k, v in weighted_metrics.items():
    print(f"{k}: {v:.6f}")

# Added standard R² calculation (based on total SSE/SST)
global_r2 = 1 - (weighted_metrics["SSE"] / weighted_metrics["SST"])
print(f"\n Global R^2 (based on SSE/SST): {global_r2:.6f}")

