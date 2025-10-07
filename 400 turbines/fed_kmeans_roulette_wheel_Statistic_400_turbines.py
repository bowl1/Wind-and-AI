import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import random
from sklearn.manifold import TSNE


# ===== Client Class =====
class client_FKM:
    def __init__(self, data, n_clusters, n_iter=1):
        self.data = data
        self.n_clusters = min(n_clusters, len(data))
        self.n_iter = n_iter
        self.means = None

    def compute_d2(self, current_centers):
        dists = np.min(
            np.linalg.norm(self.data[:, None] - current_centers[None, :], axis=2) ** 2,
            axis=1,
        )
        return dists

    def sample_point_by_d2(self, current_centers):
        d2 = self.compute_d2(current_centers)
        probs = d2 / np.sum(d2)
        idx = np.random.choice(len(self.data), p=probs)
        return self.data[idx]

    def det_local_clusters(self):
        km = KMeans(
            n_clusters=self.n_clusters, init=self.means, n_init=1, max_iter=self.n_iter
        )
        km.fit(self.data)
        self.means = km.cluster_centers_
        sample_amts = np.array(
            [np.sum(km.labels_ == i) for i in range(self.n_clusters)]
        )
        return self.means.copy(), sample_amts


# ===== Server Class =====
class server_FKM:
    def __init__(self, n_global):
        self.n_global = n_global

    def aggregate(self, local_clusters, cluster_sizes):
        kmeans = KMeans(n_clusters=self.n_global)
        kmeans.fit(local_clusters, sample_weight=cluster_sizes)
        return kmeans.cluster_centers_


# ===== DRS Initialization =====
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


# ===== Assign Labels Function =====
def assign_labels_by_centers(X, centers):
    dists = np.linalg.norm(X[:, None] - centers[None, :], axis=2)
    return np.argmin(dists, axis=1)


# ===== MAIN WORKFLOW =====
def run_federated_kmeans(
    feature_csv,
    coord_csv,
    n_clients,
    k_global,
    crounds,
    output_prefix=None,
    save_outputs=True,
):
    # Set random seeds for reproducibility
    np.random.seed(42)
    random.seed(42)

    # Output prefix automatic naming
    if output_prefix is None:
        output_prefix = f"fedkmeans_n{n_clients}_k{k_global}_r{crounds}"

    # Load Data
    df = pd.read_csv(feature_csv)
    geo_df = pd.read_csv(coord_csv)[["GSRN", "UTM_x", "UTM_y"]]

    feature_cols = [
        "mean_power_scaled",
        "std_power_scaled",
        "cv_scaled",
        "zero_ratio",
        "ramp_mean_scaled",
        "ramp_std_scaled",
    ]

    X = df[feature_cols].values
    gsrn = df["GSRN"].values

    unique_gsrn_df = df[["GSRN"]].drop_duplicates()
    merged_df = pd.merge(unique_gsrn_df, geo_df, on="GSRN", how="inner")

    geo_kmeans = KMeans(n_clusters=n_clients, random_state=42)
    merged_df["geo_cluster"] = geo_kmeans.fit_predict(merged_df[["UTM_x", "UTM_y"]])

    client_gsrn_groups = [
        merged_df[merged_df["geo_cluster"] == i]["GSRN"].tolist()
        for i in sorted(merged_df["geo_cluster"].unique())
    ]

    clients = []
    for group in client_gsrn_groups:
        client_data = X[df["GSRN"].isin(group)]
        unique_data = np.unique(client_data, axis=0)
        client_k = min(k_global, len(unique_data))
        # client_k = min(k_global, len(client_data), 4)
        if client_k >= 2:
            clients.append(client_FKM(data=unique_data, n_clusters=client_k, n_iter=1))

    print(f"\nValid clients: {len(clients)}")

    # Federated Training
    global_centers = double_roulette_init(clients, k_global)

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

    # Assign Labels + Save Results
    labels = assign_labels_by_centers(X, global_centers)
    df["cluster"] = labels

    sil_score = silhouette_score(X, labels)
    print(f"\nSilhouette Score: {sil_score:.4f}")

    if save_outputs:
        df[["GSRN", "cluster"]].to_csv(f"{output_prefix}_assignment.csv", index=False)

    # # Using t-SNE instead of PCA for better separation in 2D/3D visualization, especially for non-linear structures
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    X_tsne = tsne.fit_transform(X)

    if save_outputs:
        plt.figure(figsize=(10, 7))
        ax = sns.scatterplot(
            x=X_tsne[:, 0], y=X_tsne[:, 1], hue=labels, palette="tab10", s=100
        )
        for i, gsrn_id in enumerate(gsrn):
            ax.text(
                X_tsne[i, 0] + 0.05, X_tsne[i, 1] + 0.05, str(gsrn_id)[-4:], fontsize=8
            )

        plt.title(
            f"t-SNE (Clients={n_clients}, Clusters={k_global}, Rounds={crounds}, Silhouette={sil_score:.4f})"
        )
        plt.tight_layout()
        plt.savefig(f"{output_prefix}_tsne.png", dpi=300)
        plt.close()

    # Summary
    cluster_means = df.groupby("cluster")[feature_cols].mean().round(3)
    cluster_stds = df.groupby("cluster")[feature_cols].std().round(3)

    print("\n=== Cluster Means ===")
    print(cluster_means)
    print("\n=== Cluster STDs ===")
    print(cluster_stds)

    # Radar Plot
    # angles = np.linspace(0, 2 * np.pi, len(feature_cols), endpoint=False).tolist()
    # angles += angles[:1]

    # plt.figure(figsize=(8, 6))
    # for cluster_id in cluster_means.index:
    #     values = cluster_means.loc[cluster_id].tolist()
    #     values += values[:1]
    #     plt.polar(angles, values, label=f"Cluster {cluster_id}", marker='o')
    # plt.xticks(angles[:-1], feature_cols, fontsize=9)
    # plt.title("Cluster Feature Profile (Mean)", size=14)
    # plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.05))
    # plt.tight_layout()
    # plt.savefig(f"{output_prefix}_radar.png", dpi=300)
    # plt.show()

    return sil_score

# ===== Grid Search for Best Parameters =====
def grid_search_federated_kmeans():
    feature_csv = (
        # first try the full set of 400 turbines
        # "./selected feature statistics/features_in_group_stage_400_turbines.csv" 
        
        # second time to group the biggst cluster (remove those outliner groups and small groups)
        # "./group_stage_results/second_group/features_cluster_0_1st.csv"
        
        # third time to group the biggest cluster
        "./group_stage_results/third_group/features_cluster_0_2th.csv"
        
      
    )
    coord_csv = "./selected turbines/selected_400_turbines_filtered.csv"

    best_score = -1
    best_params = None
    results = []

    for n_clients in range(5, 15):
        for k_global in range(2, 8):
            for crounds in range(3, 10):
                print(
                    f"\n>>> Running: n_clients={n_clients}, k_global={k_global}, crounds={crounds}"
                )
                try:
                    prefix = f"auto_n{n_clients}_k{k_global}_r{crounds}"
                    score = run_federated_kmeans(
                        feature_csv=feature_csv,
                        coord_csv=coord_csv,
                        n_clients=n_clients,
                        k_global=k_global,
                        crounds=crounds,
                        output_prefix=prefix,
                        save_outputs=False,
                    )

                    results.append(
                        {
                            "n_clients": n_clients,
                            "k_global": k_global,
                            "crounds": crounds,
                            "silhouette": score,
                        }
                    )

                    if score >= best_score:
                        best_score = score
                        best_params = (n_clients, k_global, crounds)
                    elif score == best_score:
                        best_params.append(
                            (n_clients, k_global, crounds)
                        )  

                except Exception as e:
                    print(f" Failed for n={n_clients}, k={k_global}, r={crounds}: {e}")
                    continue

    # save all results to CSV
    pd.DataFrame(results).to_csv("grid_search_results.csv", index=False)
    print("\nSaved all scores to grid_search_results.csv")

    # if no successful run, raise error
    if best_params is None:
        raise RuntimeError(" No successful run found during grid search.")

    print(
        f"\nBest Silhouette Score: {best_score:.4f} with n_clients={best_params[0]}, "
        f"k_global={best_params[1]}, crounds={best_params[2]}"
    )

    # save final results for best config
    print("\n Saving final results (PCA, CSV) for best config...")

    run_federated_kmeans(
        feature_csv=feature_csv,
        coord_csv=coord_csv,
        n_clients=best_params[0],
        k_global=best_params[1],
        crounds=best_params[2],
        output_prefix=f"best_n{best_params[0]}_k{best_params[1]}_r{best_params[2]}",
        save_outputs=True,
    )


# ===== Run with Your Parameters =====
if __name__ == "__main__":
    grid_search_federated_kmeans()

# first stage group: the process takes about 4m on MacBook Pro M3
# second stage group: the process takes about 4m on MacBook Pro M3
