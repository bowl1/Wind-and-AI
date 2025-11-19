import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from collections import deque
import random

# === Reproducibility ===
SEED = 42
random.seed(SEED)
np.random.seed(SEED)


# ===== Client Class =====
class client_FKM:
    def __init__(self, data, n_clusters, n_init=1):
        """
        data: local data points (numpy array)
        n_clusters: number of local clusters
        n_init: number of initializations for KMeans
        """
        self.data = data
        self.n_clusters = min(n_clusters, len(data))  # in case data points < n_clusters
        self.means = None
        self.n_init = n_init

    def det_local_clusters(self):
        """
        Perform KMeans on local data and return cluster centers and their sizes.
        """
        km = KMeans(n_clusters=self.n_clusters, init=self.means, n_init=self.n_init)
        km.fit(self.data)
        self.means = km.cluster_centers_
        sample_amts = np.array(
            [np.sum(km.labels_ == i) for i in range(self.n_clusters)]
        )
        return self.means.copy(), sample_amts


# ===== Server Class =====
class server_FKM:
    def __init__(self, n_global):
        """
        n_global: number of global clusters
        """
        self.n_global = n_global

    def aggregate(self, local_clusters, cluster_sizes):
        """
        Aggregate local cluster centers into global centers using weighted KMeans.
        """
        kmeans = KMeans(n_clusters=self.n_global)
        kmeans.fit(local_clusters, sample_weight=cluster_sizes)
        return kmeans.cluster_centers_



# ==== Assign labels based on nearest center =====
def assign_labels_by_centers(X, centers):
    dists = np.linalg.norm(X[:, None] - centers[None, :], axis=2)
    return np.argmin(dists, axis=1)


# === Main Federated KMeans Function ===

# ===== Single Federated KMeans =====
def run_single_federated_kmeans(X_sub, n_clients, k_global, crounds):
    #  randomly split data into clients
    shuffled_indices = np.random.permutation(len(X_sub))
    groups = np.array_split(shuffled_indices, n_clients)
    client_groups = [X_sub[g] for g in groups if len(g) >= 2]

    # initialization for each client
    clients = [
        client_FKM(data=group, n_clusters=min(k_global, len(group)))
        for group in client_groups
    ]
    if len(clients) == 0:
        return np.zeros(len(X_sub)), None

    #  initialization（k-means++）
    all_data = np.vstack([client.data for client in clients])
    kmeans_init = KMeans(n_clusters=k_global, init='k-means++', n_init=1, random_state=42)
    kmeans_init.fit(all_data)
    global_centers = kmeans_init.cluster_centers_

    #  federated training rounds
    for _ in range(crounds):
        local_clusters, cluster_sizes = [], []
        for client in clients:
            client.means = global_centers[
                np.random.choice(len(global_centers), client.n_clusters, replace=False)
            ]
            centers, sizes = client.det_local_clusters()
            local_clusters.append(centers)
            cluster_sizes.append(sizes)

        global_centers = server_FKM(k_global).aggregate(
            np.vstack(local_clusters), np.concatenate(cluster_sizes)
        )

    # Finally assign labels
    labels = assign_labels_by_centers(X_sub, global_centers)
    return labels, global_centers


# ===== Dynamic Recursive Splitting (Search for Optimal Parameters at Each Layer) =====
def recursive_federated_split_dynamic(
    X, param_grid, min_silhouette=0.45, min_ratio=0.3, max_single_cluster_ratio=0.7
):
    cluster_tree = {}
    split_stats = {}
    used_params = {}
    queue = deque()
    node_counter = 0

    def init_node(idx, depth):
        nonlocal node_counter
        node_id = f"cluster_{node_counter}"
        node_counter += 1
        return {
            "id": node_id,
            "data_idx": idx,
            "size": len(idx),
            "depth": depth,
            "children": [],
            "is_outlier": False,
        }

    root_idx = np.arange(len(X))
    root = init_node(root_idx, depth=1)
    cluster_tree[root["id"]] = root
    queue.append(root)

    while queue:
        node = queue.popleft()
        idx = node["data_idx"]
        depth = node["depth"]
        subX = X[idx]

        #  if the cluster is too small, mark as outlier and skip
        if len(idx) <= len(X) * min_ratio:
            node["is_outlier"] = True
            continue

        best_score = -1
        best_labels = None
        best_params = None

        # Explore parameter combinations
        for n, k, c in param_grid:
            try:
                labels, _ = run_single_federated_kmeans(subX, n, k, c)
                if len(np.unique(labels)) <= 1:
                    continue
                score = silhouette_score(subX, labels)
                if score > best_score:
                    best_score = score
                    best_labels = labels
                    best_params = (n, k, c)
            except Exception:
                continue

        #  determine whether to split, the conditions are:
        #  1. silhouette score >= threshold
        #  2. or the cluster is too large (forced split)
        is_large_cluster = len(idx) > len(X) * max_single_cluster_ratio
        should_split = (best_labels is not None) and (
            best_score >= min_silhouette or is_large_cluster
        )

        if not should_split:
            continue

        unique_labels = np.unique(best_labels)

        # Save node information
        node["silhouette"] = best_score
        node["params"] = best_params
        node["forced_split"] = is_large_cluster

        used_params[node["id"]] = {
            "depth": depth,
            "silhouette": best_score,
            "n_clients": best_params[0],
            "k_global": best_params[1],
            "crounds": best_params[2],
            "forced_split": is_large_cluster,
        }

        if depth + 1 not in split_stats:
            split_stats[depth + 1] = []

        #  Create child clusters
        for lbl in unique_labels:
            child_idx = idx[best_labels == lbl]
            cluster_size = len(child_idx)
            split_stats[depth + 1].append(cluster_size)

            child_node = init_node(child_idx, depth + 1)
            cluster_tree[child_node["id"]] = child_node
            node["children"].append(child_node)

            if cluster_size > len(X) * min_ratio:
                queue.append(child_node)
            else:
                child_node["is_outlier"] = True

    return cluster_tree, split_stats, used_params
