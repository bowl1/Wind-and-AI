import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns


class client_FKM:
    def __init__(self, data, n_clusters, n_iter=1):
        self.data = data
        self.n_clusters = min(n_clusters, len(data))
        self.n_iter = n_iter
        self.means = None

    def compute_d2(self, current_centers):
        dists = np.min(np.linalg.norm(self.data[:, None] - current_centers[None, :], axis=2) ** 2, axis=1)
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


class server_FKM:
    def __init__(self, n_global):
        self.n_global = n_global

    def aggregate(self, local_clusters, cluster_sizes):
        kmeans = KMeans(n_clusters=self.n_global)
        kmeans.fit(local_clusters, sample_weight=cluster_sizes)
        return kmeans.cluster_centers_


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


def assign_labels_by_centers(X, centers):
    dists = np.linalg.norm(X[:, None] - centers[None, :], axis=2)
    return np.argmin(dists, axis=1)


def run_federated_kmeans(
    stats_path,
    powercurve_path,
    coord_csv,
    n_clients,
    k_global,
    crounds,
    output_prefix=None,
    save_outputs=True,
):
    # ==== Load and Merge features ====
    stats_df = pd.read_csv(stats_path)
    power_df = pd.read_csv(powercurve_path)[[
        "GSRN", "residual_mse", "fft_0", "fft_1", "fft_2", "fft_3", "fft_4"
    ]]

    df = pd.merge(stats_df, power_df, on="GSRN", how="inner")

    feature_cols = [
        "mean_power_scaled", "std_power_scaled", "cv_scaled", "zero_ratio",
        "ramp_mean_scaled", "ramp_std_scaled",
         "fft_0", "fft_1", "fft_2", "fft_3", "fft_4"
    ]
    X = df[feature_cols].values
    X_scaled = StandardScaler().fit_transform(X)
    gsrn = df["GSRN"].values

    geo_df = pd.read_csv(coord_csv)[["GSRN", "UTM_x", "UTM_y"]]
    merged_df = pd.merge(df[["GSRN"]].drop_duplicates(), geo_df, on="GSRN", how="inner")
    geo_kmeans = KMeans(n_clusters=n_clients, random_state=42)
    merged_df["geo_cluster"] = geo_kmeans.fit_predict(merged_df[["UTM_x", "UTM_y"]])

    client_gsrn_groups = [
        merged_df[merged_df["geo_cluster"] == i]["GSRN"].tolist()
        for i in sorted(merged_df["geo_cluster"].unique())
    ]

    clients = []
    for group in client_gsrn_groups:
        client_data = X_scaled[df["GSRN"].isin(group)]
        client_k = min(k_global, len(client_data), 4)
        if client_k >= 2:
            clients.append(client_FKM(data=client_data, n_clusters=client_k, n_iter=1))

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

        global_centers = server_FKM(n_global=k_global).aggregate(
            np.vstack(local_clusters), np.concatenate(cluster_sizes)
        )

    labels = assign_labels_by_centers(X_scaled, global_centers)
    df["cluster"] = labels
    sil_score = silhouette_score(X_scaled, labels)
    print(f"\nSilhouette Score: {sil_score:.4f}")

    if save_outputs:
        if output_prefix is None:
            output_prefix = f"fedkmeans_n{n_clients}_k{k_global}_r{crounds}"

        df[["GSRN", "cluster"]].to_csv(f"{output_prefix}_assignment.csv", index=False)

        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)
        plt.figure(figsize=(10, 7))
        ax = sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=labels, palette="tab10", s=100)
        for i, gsrn_id in enumerate(gsrn):
            ax.text(X_pca[i, 0]+0.05, X_pca[i, 1]+0.05, str(gsrn_id)[-4:], fontsize=8)
        plt.title(f"PCA (n_clients={n_clients}, k_global={k_global}, crounds={crounds}, Silhouette={sil_score:.2f})")
        plt.tight_layout()
        plt.savefig(f"{output_prefix}_pca.png", dpi=300)
        plt.close()

    return sil_score


def grid_search_federated_kmeans(stats_path, powercurve_path, coord_csv):
    best_score = -1
    best_params = None
    results = []

    for n_clients in range(5, 15):
        for k_global in range(4, 8):
            for crounds in range(3, 10):
                print(f"\n>>> Running: n_clients={n_clients}, k_global={k_global}, crounds={crounds}")
                try:
                    prefix = f"auto_n{n_clients}_k{k_global}_r{crounds}"
                    score = run_federated_kmeans(
                        stats_path=stats_path,
                        powercurve_path=powercurve_path,
                        coord_csv=coord_csv,
                        n_clients=n_clients,
                        k_global=k_global,
                        crounds=crounds,
                        output_prefix=prefix,
                        save_outputs=False,
                    )

                    results.append({
                        "n_clients": n_clients,
                        "k_global": k_global,
                        "crounds": crounds,
                        "silhouette": score,
                    })

                    if score > best_score:
                        best_score = score
                        best_params = (n_clients, k_global, crounds)

                except Exception as e:
                    print(f"Failed for n={n_clients}, k={k_global}, r={crounds}: {e}")

    pd.DataFrame(results).to_csv("grid_search_results.csv", index=False)
    print("\nSaved grid search results to grid_search_results.csv")

    if best_params is None:
        raise RuntimeError("No successful run found during grid search.")

    print(f"\nBest Silhouette Score: {best_score:.4f} with n_clients={best_params[0]}, k_global={best_params[1]}, crounds={best_params[2]}")

    run_federated_kmeans(
        stats_path=stats_path,
        powercurve_path=powercurve_path,
        coord_csv=coord_csv,
        n_clients=best_params[0],
        k_global=best_params[1],
        crounds=best_params[2],
        output_prefix=f"best_n{best_params[0]}_k{best_params[1]}_r{best_params[2]}",
        save_outputs=True,
    )


if __name__ == "__main__":
    grid_search_federated_kmeans(
        stats_path="./selected feature statistics/features_in_group_stage_400_turbines.csv",
        powercurve_path="./selected feature statistics/power_curve_features_in_group_stage_400_turbines.csv",
        coord_csv="./selected turbines/selected_400_turbines_filtered.csv"
    )