import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans


# ===== Client =====
class client_FKM:
    def __init__(self, data, n_clusters, n_iter=1):
        self.data = data
        self.n_clusters = min(n_clusters, len(data))  # 避免样本数小于聚类数
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


# ===== DRS Initialization Function (Double Roulette Sampling) =====
def double_roulette_init(clients, k):
    # Step 1: Select the first center randomly from all clients
    first_client = np.random.choice(clients)
    first_idx = np.random.randint(len(first_client.data))
    centers = [first_client.data[first_idx]]

    for _ in range(1, k):
        # Step 2: Calculate the probability for each client
        d2_sums = [np.sum(c.compute_d2(np.array(centers))) for c in clients]
        probs_client = d2_sums / np.sum(d2_sums)

        # Step 3: The first round of roulette (select a client)
        selected_client = np.random.choice(clients, p=probs_client)

        # Step 4: The second round of roulette (select a point within the client)
        new_center = selected_client.sample_point_by_d2(np.array(centers))
        centers.append(new_center)

    return np.array(centers)


# ===== Main Program Entry =====

# ==== Load Data and Standardize ====
# 1. Read two feature files
df_stats = pd.read_csv("features_in_group_stage_50_turbines.csv")
df_powercurve = pd.read_csv("power_curve_features_in_group_stage_50_turbines.csv")

# 2. Merge the two files (on GSRN)
df = pd.merge(df_stats, df_powercurve[["GSRN", "residual_mse", "fft_0", "fft_1", "fft_2", "fft_3", "fft_4"]], on="GSRN", how="inner")

# 3. Separate GSRN and feature columns
gsrn = df["GSRN"].values
X = df.drop(columns=["GSRN"]).values

# 4. Standardize
X_scaled = StandardScaler().fit_transform(X)
# # ==== Group into clients (every 5 or 3 GSRN combine into one client) ====

# 1. Acquire unique GSRN
unique_gsrn_df = df[["GSRN"]].drop_duplicates()

# 2. Load real coordinates (from master data)
geo_df = pd.read_csv("selected_50_turbines_filtered.csv")[["GSRN", "UTM_x", "UTM_y"]]

# 3. Merge to get the locations of the turbines to be clustered
merged_df = pd.merge(unique_gsrn_df, geo_df, on="GSRN", how="inner")

# 4. Geo-based clustering to form initial client groups
n_clients = 8 # the number of clients
k_global = 3 # the global cluster number
geo_kmeans = KMeans(n_clusters=n_clients, random_state=42)
merged_df["geo_cluster"] = geo_kmeans.fit_predict(merged_df[["UTM_x", "UTM_y"]])

# 5. Construct client_gsrn_groups (no duplicates)
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

print(f"\n valid clients quantity: {len(clients)}")

# ==== Federated Training Process ====
crounds = 5 # the communication rounds
global_centers = double_roulette_init(clients, k_global)

for _ in range(crounds):
    local_clusters, cluster_sizes = [], []
    for client in clients:
        # assign global centers to client (randomly sample if not enough)
        if len(global_centers) >= client.n_clusters:
            client.means = global_centers[np.random.choice(len(global_centers), client.n_clusters, replace=False)]
        else:
            client.means = global_centers[:client.n_clusters]  # fallback

        centers, sizes = client.det_local_clusters()
        local_clusters.append(centers)
        cluster_sizes.append(sizes)

    local_clusters = np.vstack(local_clusters)
    cluster_sizes = np.concatenate(cluster_sizes)
    server = server_FKM(n_global=k_global)
    global_centers = server.aggregate(local_clusters, cluster_sizes)

# ==== Clustering Labels + PCA Visualization ====
def assign_labels_by_centers(X, centers):
    dists = np.linalg.norm(X[:, None] - centers[None, :], axis=2)
    return np.argmin(dists, axis=1)

labels = assign_labels_by_centers(X_scaled, global_centers)
df["cluster"] = labels  # 把聚类结果写入原始 dataframe

sil_score = silhouette_score(X_scaled, labels)
print(f"\n Silhouette Score: {sil_score:.4f}")

# ========== output each cluster's GSRN ==========
grouped = df[["GSRN", "cluster"]].drop_duplicates().groupby("cluster")["GSRN"].apply(list)

print("\n final group of GSRN:")
for cluster_id, gsrns in grouped.items():
    print(f"  Cluster {cluster_id}: {gsrns}")

# ==== Visualization ====
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(10, 7))
ax = sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=labels, palette="tab10", s=100)

# Add the last 4 digits of each point's GSRN as a label
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


# === Generate Combined Group Comparison CSV ===
group_records = []

# —— add Geo-based Initial Groups ——
for i, group in enumerate(client_gsrn_groups):
    group_records.append({
        "GroupName": f"GeoClient {i+1}",
        "Type": "Initial",
        "GSRNs": ", ".join(str(g) for g in group)
    })

# —— add Final Clusters ——
final_grouped = df[["GSRN", "cluster"]].drop_duplicates().groupby("cluster")["GSRN"].apply(list)

for cluster_id, gsrns in final_grouped.items():
    group_records.append({
        "GroupName": f"Cluster {cluster_id}",
        "Type": "Final",
        "GSRNs": ", ".join(str(g) for g in gsrns)
    })

# —— save to a CSV file ——
combined_df = pd.DataFrame(group_records)
combined_filename = f"group_comparison_n{n_clients}_k{k_global}_r{crounds}.csv"
combined_df.to_csv(combined_filename, index=False)
print(f"\nCombined group (initial + final) saved to file: {combined_filename}")