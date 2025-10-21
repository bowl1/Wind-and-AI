import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns

# ===== Client Class =====
class client_FKM:
    def __init__(self, data, n_clusters, n_iter=1):
        self.data = data
        self.n_clusters = min(n_clusters, len(data))
        self.n_iter = n_iter
        self.means = None

    def compute_d2(self, current_centers):
        dists = np.min(np.linalg.norm(self.data[:, None] - current_centers[None, :], axis=2)**2, axis=1)
        return dists

    def sample_point_by_d2(self, current_centers):
        d2 = self.compute_d2(current_centers)
        probs = d2 / np.sum(d2)
        idx = np.random.choice(len(self.data), p=probs)
        return self.data[idx]

    def det_local_clusters(self):
        km = KMeans(n_clusters=self.n_clusters, init=self.means, n_init=1, max_iter=self.n_iter)
        km.fit(self.data)
        self.means = km.cluster_centers_
        sample_amts = np.array([np.sum(km.labels_ == i) for i in range(self.n_clusters)])
        return self.means.copy(), sample_amts

# ===== Server Aggregation Class =====
class server_FKM:
    def __init__(self, n_global):
        self.n_global = n_global

    def aggregate(self, local_clusters, cluster_sizes):
        kmeans = KMeans(n_clusters=self.n_global)
        kmeans.fit(local_clusters, sample_weight=cluster_sizes)
        return kmeans.cluster_centers_

# ===== Double Roulette Sampling Initialization =====
def double_roulette_init(clients, k):
    first_client = np.random.choice(clients)
    first_idx = np.random.randint(len(first_client.data))
    centers = [first_client.data[first_idx]]

    for _ in range(1, k):
        d2_sums = [np.sum(c.compute_d2(np.array(centers))) for c in clients]
        probs_client = d2_sums / np.sum(d2_sums)
        selected_client = np.random.choice(clients, p=probs_client)
        new_center = selected_client.sample_point_by_d2(np.array(centers))
        centers.append(new_center)

    return np.array(centers)

# ===== Main Execution =====
# Load statistical and FFT data
df_stats = pd.read_csv("features_in_group_stage.csv")
df_powercurve = pd.read_csv("power_curve_features_in_group_stage.csv")

# Merge both datasets by GSRN
df = pd.merge(df_stats, df_powercurve[["GSRN", "fft_0", "fft_1", "fft_2", "fft_3", "fft_4"]], on="GSRN", how="inner")

# Apply PCA to compress FFT features
fft_cols = ["fft_0", "fft_1", "fft_2", "fft_3", "fft_4"]
pca_fft = PCA(n_components=1)
df["fft_compressed"] = pca_fft.fit_transform(df[fft_cols])

# Feature columns used for clustering
feature_cols = [
    "mean_power_scaled",
    "std_power_scaled",
    "cv_scaled",
    "zero_ratio",
    "ramp_mean_scaled",
    "ramp_std_scaled",
    "fft_compressed"
]

X = df[feature_cols].values
X_scaled = StandardScaler().fit_transform(X)
gsrn = df["GSRN"].values

# Load location info and create geo-based client groups
unique_gsrn_df = df[["GSRN"]].drop_duplicates()
geo_df = pd.read_csv("selected_15_turbines_filtered.csv")[["GSRN", "UTM_x", "UTM_y"]]
merged_df = pd.merge(unique_gsrn_df, geo_df, on="GSRN", how="inner")

n_clients = 3
k_global = 3
geo_kmeans = KMeans(n_clusters=n_clients, random_state=42)
merged_df["geo_cluster"] = geo_kmeans.fit_predict(merged_df[["UTM_x", "UTM_y"]])

client_gsrn_groups = []
for cluster_id in sorted(merged_df["geo_cluster"].unique()):
    group = merged_df[merged_df["geo_cluster"] == cluster_id]["GSRN"].tolist()
    client_gsrn_groups.append(group)

clients = []
print("Client initial group:")
for idx, group in enumerate(client_gsrn_groups):
    print(f"  Client {idx+1}: {[str(g) for g in group]}")
    client_data = X_scaled[df["GSRN"].isin(group)]
    client_k = min(k_global, len(client_data), 4)
    if client_k >= 2:
        clients.append(client_FKM(data=client_data, n_clusters=client_k, n_iter=1))

print(f"\nValid clients: {len(clients)}")

# Federated Clustering Training
crounds = 4
global_centers = double_roulette_init(clients, k_global)

for _ in range(crounds):
    local_clusters, cluster_sizes = [], []
    for client in clients:
        client.means = global_centers[np.random.choice(len(global_centers), client.n_clusters, replace=False)]
        centers, sizes = client.det_local_clusters()
        local_clusters.append(centers)
        cluster_sizes.append(sizes)

    local_clusters = np.vstack(local_clusters)
    cluster_sizes = np.concatenate(cluster_sizes)
    server = server_FKM(n_global=k_global)
    global_centers = server.aggregate(local_clusters, cluster_sizes)

# Assign labels and visualize with PCA
def assign_labels_by_centers(X, centers):
    dists = np.linalg.norm(X[:, None] - centers[None, :], axis=2)
    return np.argmin(dists, axis=1)

labels = assign_labels_by_centers(X_scaled, global_centers)
df["cluster"] = labels
sil_score = silhouette_score(X_scaled, labels)
print(f"\nSilhouette Score: {sil_score:.4f}")

# Output cluster membership
print("\nFinal group of GSRNs:")
grouped = df[["GSRN", "cluster"]].drop_duplicates().groupby("cluster")["GSRN"].apply(list)
for cluster_id, gsrns in grouped.items():
    print(f"  Cluster {cluster_id}: {gsrns}")

# PCA visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
plt.figure(figsize=(10, 7))
ax = sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=labels, palette="tab10", s=100)
for i, gsrn_id in enumerate(gsrn):
    ax.text(X_pca[i, 0]+0.05, X_pca[i, 1]+0.05, str(gsrn_id)[-4:], fontsize=8)
plt.title(f"PCA with Federated KMeans + DRS (Clients: {n_clients}, Clusters: {k_global}, Rounds: {crounds}, Silhouette: {sil_score:.2f})")
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.grid(True)
plt.legend(title="Cluster")
plt.tight_layout()
plt.savefig(f"federated_kmeans_pca_n{n_clients}_k{k_global}_r{crounds}.png", dpi=300)
plt.show()

# Save Group Comparison CSV
group_records = []
for i, group in enumerate(client_gsrn_groups):
    group_records.append({"GroupName": f"GeoClient {i+1}", "Type": "Initial", "GSRNs": ", ".join(str(g) for g in group)})
for cluster_id, gsrns in grouped.items():
    group_records.append({"GroupName": f"Cluster {cluster_id}", "Type": "Final", "GSRNs": ", ".join(str(g) for g in gsrns)})
combined_df = pd.DataFrame(group_records)
combined_filename = f"group_comparison_n{n_clients}_k{k_global}_r{crounds}.csv"
combined_df.to_csv(combined_filename, index=False)
print(f"\nGroup comparison saved to: {combined_filename}")