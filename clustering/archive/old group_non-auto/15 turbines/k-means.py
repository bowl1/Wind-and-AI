import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

# ==== Load Data ====
df = pd.read_csv("features_in_group_stage.csv")

# ==== Define Feature Columns ====
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

# ==== KMeans Clustering ====
k = 5 # try 3-5
kmeans = KMeans(n_clusters=k, random_state=42)
labels = kmeans.fit_predict(X)
df["cluster"] = labels

# ==== Silhouette Score ====
sil_score = silhouette_score(X, labels)
print(f"Silhouette Score: {sil_score:.4f}")

# ==== PCA Visualization ====
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

plt.figure(figsize=(10, 7))
ax = sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=labels, palette="tab10", s=100)

for i, gsrn_id in enumerate(gsrn):
    ax.text(X_pca[i, 0]+0.05, X_pca[i, 1]+0.05, str(gsrn_id)[-4:], fontsize=8)

plt.title(f"KMeans Clustering PCA (k={k}, Silhouette: {sil_score:.2f})")
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.grid(True)
plt.legend(title="Cluster")
plt.tight_layout()
plt.savefig(f"kmeans_pca_k{k}.png", dpi=300)
plt.show()

# ==== Cluster Summary ====
cluster_means = df.groupby("cluster")[feature_cols].mean().round(3)
cluster_stds = df.groupby("cluster")[feature_cols].std().round(3)
print("\n=== Cluster Means ===")
print(cluster_means)
print("\n=== Cluster STDs ===")
print(cluster_stds)

# ==== Save results ====
df[["GSRN", "cluster"]].to_csv(f"kmeans_cluster_assignment_k{k}.csv", index=False)