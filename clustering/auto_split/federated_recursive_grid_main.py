import pandas as pd
import numpy as np
# from cluster_tree_utils import recursive_federated_split_dynamic
from cluster_tree_kmeans_utils import recursive_federated_split_dynamic
from plot import plot_tsne_all_clusters, plot_pca_3d, extract_leaf_labels

# === Step 1: Load filtered GSRN + coords ===
filtered_df = pd.read_csv(
    #"./200 turbines/selected turbines/selected_200_turbines_filtered.csv"
    #"./50 turbines/selected turbines/selected_50_turbines_filtered.csv"
     "./400 turbines/selected turbines/selected_400_turbines_filtered.csv"
    #"./15 turbines/selected_turbines/selected_15_turbines_filtered.csv"
)
filtered_gsrn = set(filtered_df["GSRN"])
coords_df = filtered_df[["GSRN", "UTM_x", "UTM_y"]]

# === Step 2: Load feature data ===
feature_df = pd.read_csv(
    #"./200 turbines/selected feature statistics/features_in_group_stage_200_turbines.csv"
    #"./50 turbines/selected feature statistics/features_in_group_stage_50_turbines.csv"
    "./400 turbines/selected feature statistics/features_in_group_stage_400_turbines.csv"
   # "./15 turbines/selected_feature_statistics/features_in_group_stage.csv"
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


# === Step 5: Build param grid ===
param_grid = [
    (n, k, c) for n in range(3, 10) for k in range(3, 11) for c in range(3, 11)
]

# === Step 6: Run Recursive Federated Clustering ===
cluster_tree, best_split_stats, used_params = recursive_federated_split_dynamic(
    X, param_grid
)

# === Step 7: Compute silhouette summary ===
avg_score = np.mean(
    [
        node.get("silhouette", 0)
        for node in cluster_tree.values()
        if "silhouette" in node
    ]
)

# === Step 8: Save Outputs ===
split_stats_df = pd.DataFrame(
    [
        {"Layer": layer, "NumClusters": len(sizes), "Sizes": sizes}
        for layer, sizes in best_split_stats.items()
    ]
)
split_stats_df.to_csv("best_tree_layerwise_stats.csv", index=False)

used_params_df = pd.DataFrame.from_dict(used_params, orient="index").reset_index(
    names=["Node"]
)
used_params_df.to_csv("best_tree_node_params.csv", index=False)

summary_df = pd.DataFrame([{"AvgSilhouetteScore": avg_score}])
summary_df.to_csv("best_score_summary.csv", index=False)

print("\n All done. Files saved:")
print(" - best_tree_layerwise_stats.csv")
print(" - best_tree_node_params.csv")
print(" - best_score_summary.csv")

# === Step 9: Plot final t-SNE visualization ===
gsrn_list = merged_df["GSRN"].tolist()
plot_tsne_all_clusters(X, cluster_tree, gsrn_list=gsrn_list)

# === Step 10: Plot final PCA visualization ===
leaf_labels = extract_leaf_labels(X, cluster_tree)
plot_pca_3d(X, leaf_labels, gsrn_list=gsrn_list)

# === Step 11: Compute and save per-cluster feature statistics ===
# prepare leaf labels into dataframe
feature_df["Cluster"] = leaf_labels

# calculate size, mean, std for each feature per cluster
cluster_sizes = feature_df.groupby("Cluster").size().rename("Count")
cluster_means = feature_df.groupby("Cluster")[feature_cols].mean().round(4)
cluster_stds = feature_df.groupby("Cluster")[feature_cols].std().round(4)

# combine into single summary dataframe
cluster_summary = pd.concat([cluster_sizes, cluster_means.add_suffix("_mean"), cluster_stds.add_suffix("_std")], axis=1)
cluster_summary.index.name = "Cluster"

cluster_summary.to_csv("cluster_feature_statistics.csv")
print("Saved: cluster_feature_statistics.csv")

# === Step 12: Save turbine IDs (GSRNs) grouped by cluster ===
# prepare dataframe with GSRN and Cluster
cluster_assignments = pd.DataFrame({
    "GSRN": merged_df["GSRN"],
    "Cluster": leaf_labels
})

# group by cluster and aggregate GSRNs into lists
grouped_turbines = cluster_assignments.groupby("Cluster")["GSRN"].apply(list).reset_index()

grouped_turbines.to_csv("cluster_turbine_ids.csv", index=False)
print("Saved: cluster_turbine_ids.csv")
