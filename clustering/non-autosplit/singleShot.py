import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from plot import plot_tsne_and_pca_baseline

# ===== Reproducibility =====
SEED = 42
np.random.seed(SEED)

# ===== Client Class =====
class ClientFKM:
    """
    单个“联邦客户端”，执行：
    - 使用给定初始中心做一次本地 KMeans（1-2 次迭代）
    - 返回本地簇中心和每个簇的样本数
    """
    def __init__(self, data, n_clusters, n_iter=1):
        self.data = data
        self.n_clusters = min(n_clusters, len(data))
        self.n_iter = n_iter
        self.means = None  # 当前使用的初始中心

    def det_local_clusters(self):
        # 在本地数据上跑一次 KMeans（使用 self.means 作为 init）
        km = KMeans(
            n_clusters=self.n_clusters,
            init=self.means,
            n_init=1,
            max_iter=self.n_iter,
            random_state=SEED,
        )
        km.fit(self.data)
        self.means = km.cluster_centers_
        sample_amts = np.array(
            [np.sum(km.labels_ == i) for i in range(self.n_clusters)]
        )
        return self.means.copy(), sample_amts


# ===== Server Class =====
class ServerFKM:
    """
    服务器端：聚合所有客户端的本地簇中心，得到新的全局簇中心
    """
    def __init__(self, n_global):
        self.n_global = n_global

    def aggregate(self, local_clusters, cluster_sizes):
        # 用带 sample_weight 的 KMeans 来平衡各客户端贡献
        kmeans = KMeans(n_clusters=self.n_global, random_state=SEED)
        kmeans.fit(local_clusters, sample_weight=cluster_sizes)
        return kmeans.cluster_centers_


# ===== Helper: assign labels by nearest global center =====
def assign_labels_by_centers(X, centers):
    dists = np.linalg.norm(X[:, None] - centers[None, :], axis=2)
    return np.argmin(dists, axis=1)


# ===== Single-shot Federated KMeans++ Procedure =====
def run_single_shot_federated_kmeans(X, n_clients, k_global, crounds):
    """
    Single-shot KMeans++ + FL baseline:
    - 在全体数据上用 KMeans++ 初始化全局中心（一次）
    - 将数据随机划分为 n_clients 个“合成客户端”
    - 进行 crounds 轮联邦 KMeans（本地更新 + 全局聚合）
    - 最后用得到的 global_centers 为所有样本赋簇标签
    """
    N = len(X)
    if N < k_global:
        raise ValueError("样本数小于簇数，无法聚类")

    # ---- 1. 随机划分客户端（模拟联邦）----
    shuffled_idx = np.random.permutation(N)
    groups = np.array_split(shuffled_idx, n_clients)
    client_groups_idx = [g for g in groups if len(g) >= 2]

    clients = []
    for g in client_groups_idx:
        client_data = X[g]
        client_k = min(k_global, len(client_data))
        if client_k >= 2:
            clients.append(ClientFKM(data=client_data, n_clusters=client_k, n_iter=1))

    if len(clients) == 0:
        raise RuntimeError("没有有效客户端，检查 n_clients 和数据量设置。")

    # ---- 2. 全局 KMeans++ 初始化（在所有样本上）----
    kmeans_init = KMeans(
        n_clusters=k_global,
        init="k-means++",
        n_init=10,
        max_iter=300,
        random_state=SEED,
    )
    kmeans_init.fit(X)
    global_centers = kmeans_init.cluster_centers_

    # ---- 3. 联邦 KMeans 迭代 ----
    server = ServerFKM(n_global=k_global)

    for _ in range(crounds):
        local_clusters, cluster_sizes = [], []

        for client in clients:
            # 给客户端一个子集全局中心作为初始中心
            if len(global_centers) >= client.n_clusters:
                init_idx = np.random.choice(
                    len(global_centers),
                    client.n_clusters,
                    replace=False,
                )
                client.means = global_centers[init_idx]
            else:
                client.means = global_centers[: client.n_clusters]

            centers, sizes = client.det_local_clusters()
            local_clusters.append(centers)
            cluster_sizes.append(sizes)

        local_clusters = np.vstack(local_clusters)
        cluster_sizes = np.concatenate(cluster_sizes)

        # 聚合更新全局中心
        global_centers = server.aggregate(local_clusters, cluster_sizes)

    # ---- 4. 用最终 global_centers 给全体样本打标签 ----
    labels = assign_labels_by_centers(X, global_centers)
    return labels, global_centers


# ================== Main Script ==================

# === Step 1: Load filtered GSRN + coords ===
filtered_df = pd.read_csv(
    "./selected turbines/selected_400_turbines_filtered.csv"
)
filtered_gsrn = set(filtered_df["GSRN"])
coords_df = filtered_df[["GSRN", "UTM_x", "UTM_y"]]

# === Step 2: Load feature data ===
feature_df = pd.read_csv(
    "./selected feature statistics/features_in_group_stage_400_turbines.csv"
)
feature_df = feature_df[feature_df["GSRN"].isin(filtered_gsrn)].reset_index(drop=True)

# === Step 3: Merge for geo info ===
merged_df = pd.merge(feature_df, coords_df, on="GSRN", how="inner")

# === Step 4: Prepare data matrix X and labels ===
feature_cols = [
    "mean_power_scaled",
    "std_power_scaled",
    "cv_scaled",
    "zero_ratio",
    "ramp_mean_scaled",
    "ramp_std_scaled",
]
X = merged_df[feature_cols].values
gsrn_list = merged_df["GSRN"].tolist()

# ====== Single-shot KMeans++ + FL 参数 ======
N_CLIENTS = 8   # 和 Auto-split 根节点一致
K_GLOBAL = 6    # 由 elbow + silhouette 共同选择
C_ROUNDS = 9    # 和 Auto-split 根节点一致

# === Step 5: Run Single-shot Federated KMeans++ ===
labels, global_centers = run_single_shot_federated_kmeans(
    X,
    n_clients=N_CLIENTS,
    k_global=K_GLOBAL,
    crounds=C_ROUNDS,
)

sil_score = silhouette_score(X, labels)
print(f"\n[Single-shot KMeans++ + FL] Silhouette score = {sil_score:.4f}")

# === Step 6: Save silhouette summary ===
summary_df = pd.DataFrame(
    [
        {
            "Setting": "SingleShot_KMeansPP_FL",
            "n_clients": N_CLIENTS,
            "k_global": K_GLOBAL,
            "crounds": C_ROUNDS,
            "SilhouetteScore": sil_score,
        }
    ]
)
summary_df.to_csv("single_shot_summary.csv", index=False)
print("Saved: single_shot_summary.csv")

# === Step 7: Per-cluster feature statistics ===
feature_df["Cluster"] = labels

cluster_sizes = feature_df.groupby("Cluster").size().rename("Count")
cluster_means = feature_df.groupby("Cluster")[feature_cols].mean().round(4)
cluster_stds = feature_df.groupby("Cluster")[feature_cols].std().round(4)

cluster_summary = pd.concat(
    [
        cluster_sizes,
        cluster_means.add_suffix("_mean"),
        cluster_stds.add_suffix("_std"),
    ],
    axis=1,
)
cluster_summary.index.name = "Cluster"
cluster_summary.to_csv("single_shot_cluster_feature_statistics.csv")
print("Saved: single_shot_cluster_feature_statistics.csv")

# === Step 8: Save turbine IDs (GSRNs) grouped by cluster ===
cluster_assignments = pd.DataFrame(
    {
        "GSRN": merged_df["GSRN"],
        "Cluster": labels,
    }
)

grouped_turbines = (
    cluster_assignments.groupby("Cluster")["GSRN"].apply(list).reset_index()
)
grouped_turbines.to_csv("single_shot_cluster_turbine_ids.csv", index=False)
print("Saved: single_shot_cluster_turbine_ids.csv")

# === Step 9: Simple PCA visualization (可选) ===
gsrn_list = merged_df["GSRN"].tolist()  # 或 df["GSRN"]
plot_tsne_and_pca_baseline(
    X,
    labels,
    gsrn_list=gsrn_list,
    tsne_path="single_shot_tsne_clusters.png",
    pca3d_path="single_shot_pca_3d_clusters.png",
    title_prefix="Single-shot KMeans++ + FL (k=6)",
)