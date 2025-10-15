import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

# ==== Load Data ====
df = pd.read_csv("./selected feature statistics/features_in_group_stage_400_turbines.csv")

# ==== Define Feature Columns (Already Scaled) ====
feature_cols = [
    "mean_power_scaled",
    "std_power_scaled",
    "cv_scaled",
    "zero_ratio",
    "ramp_mean_scaled",
    "ramp_std_scaled"
]

X = df[feature_cols].values
gsrn = df["GSRN"].values

# ==== Auto Test k ====
k_range = range(2, 11)  # test k = 2 to 10
results = []

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init="auto")
    labels = kmeans.fit_predict(X)
    score = silhouette_score(X, labels)
    results.append((k, score))
    print(f"k={k}, silhouette score={score:.4f}")

# ==== Find Best k ====
best_k, best_score = max(results, key=lambda x: x[1])
print(f"\nBest k: {best_k} with silhouette score = {best_score:.4f}")

# ==== Run KMeans again with best k ====
kmeans = KMeans(n_clusters=best_k, random_state=42, n_init="auto")
labels = kmeans.fit_predict(X)
df["cluster"] = labels

# ==== Generate Cluster Labels with Count ====
cluster_counts = Counter(labels)
label_names = [f"Cluster {label} ({cluster_counts[label]})" for label in labels]
df["cluster_label"] = label_names

# ==== PCA Visualization ====
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

plt.figure(figsize=(10, 7))
ax = sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=df["cluster_label"], palette="tab10", s=100)

# Optional: turbine ID text
for i, gsrn_id in enumerate(gsrn):
    ax.text(X_pca[i, 0]+0.05, X_pca[i, 1]+0.05, str(gsrn_id)[-4:], fontsize=7)

# Add cluster count text near centroids
for cluster_id in cluster_counts:
    cluster_points = X_pca[np.array(labels) == cluster_id]
    centroid = cluster_points.mean(axis=0)
    ax.text(
        centroid[0], centroid[1],
        f"{cluster_id} ({cluster_counts[cluster_id]})",
        fontsize=11, fontweight="bold", color="black",
        bbox=dict(facecolor='white', alpha=0.6, edgecolor='gray', boxstyle='round,pad=0.3')
    )

plt.title(f"KMeans Clustering (k={best_k}, Silhouette={best_score:.2f})")
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.grid(True)
plt.legend(title="Cluster", loc='best')
plt.tight_layout()
plt.savefig(f"kmeans_best_k{best_k}_pca.png", dpi=300)
plt.show()

# ==== Cluster Summary ====
cluster_means = df.groupby("cluster")[feature_cols].mean().round(3)
cluster_stds = df.groupby("cluster")[feature_cols].std().round(3)
print("\n=== Cluster Means ===")
print(cluster_means)
print("\n=== Cluster STDs ===")
print(cluster_stds)

# ==== Save cluster assignments ====
df[["GSRN", "cluster"]].to_csv(f"kmeans_cluster_assignment_k{best_k}.csv", index=False)

# ==== Save all silhouette scores ====
pd.DataFrame(results, columns=["k", "silhouette_score"]).to_csv("kmeans_silhouette_scores.csv", index=False)