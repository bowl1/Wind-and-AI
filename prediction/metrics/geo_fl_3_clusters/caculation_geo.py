import re
import ast
import csv

# === 1. Manually define the number of samples for each cluster ===
SAMPLE_COUNTS = {
    "0": 143,
    "1": 158,
    "2": 99,
}

# === 2. Define auxiliary functions: clean np.float32(...) and parse into Python list ===
def safe_parse_list(s):
    """清理 np.float32(...) 和 inf 表达式，并解析成 Python 列表"""
    # If the string contains 'inf', skip parsing directly
    if "inf" in s:
        raise ValueError("includes 'inf', cannot parse safely")

    # Normal cleaning np.float32(...) syntax
    s = re.sub(r'np\.float32\((.*?)\)', r'\1', s)
    return ast.literal_eval(s)

# === 3. Read the contents of txt file ===
with open("summary_test.txt", "r") as f:
    content = f.read()

# === 4. Analyze the indicators of each cluster ===
blocks = content.split("cluster ")
metrics = {}

for block in blocks[1:]:  # skip first empty block
    lines = block.strip().splitlines()
    cluster_id = lines[0].strip(":")  # Get cluster ID
    cluster_metrics = {}
    for line in lines:
        if ":" in line and "[" in line:
            name, values_str = line.strip().split(": ", 1)
            try:
                values = safe_parse_list(values_str)
                cluster_metrics[name.strip()] = float(values[-1])  # Get the value of the last round
            except Exception as e:
                print(f"⚠️ the reason for skiping lines ({name}): {values_str}\nreason: {e}")
    metrics[cluster_id] = cluster_metrics
    print(f"Cluster {cluster_id} metrics (last round): {cluster_metrics}")

# === 5. Calculate weighted average indicator ===
total_samples = sum(SAMPLE_COUNTS.values())
weighted_metrics = {k: 0 for k in ["MSE", "RMSE", "MAE", "NRMSE", "SSE", "SST"]}

for cid, metric in metrics.items():
    count = SAMPLE_COUNTS.get(cid, 0)
    for k in weighted_metrics:
        if k in ["SSE", "SST"]:
            weighted_metrics[k] += metric.get(k, 0.0)
        else:
            weighted_metrics[k] += metric.get(k, 0.0) * count / total_samples

# === 6. Output summary indicators ===
print("\n📊 Final Weighted Global Metrics (last round):")
for k, v in weighted_metrics.items():
    print(f"{k}: {v:.6f}")

# === 7. Calculation standard R² ===
global_r2 = 1 - (weighted_metrics["SSE"] / weighted_metrics["SST"])
print(f"\nGlobal R^2 (based on total SSE/SST): {global_r2:.6f}")

# === 8. Save as CSV file ===
with open("geo_test_summary.csv", "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["Metric", "WeightedValue"])
    for k, v in weighted_metrics.items():
        writer.writerow([k, v])
    writer.writerow(["Global_R2", global_r2])

print("\n Saved results to geo_test_summary.csv")