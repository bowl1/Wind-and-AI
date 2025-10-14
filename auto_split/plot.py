
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from itertools import product
from matplotlib.cm import get_cmap
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import random

# === Reproducibility ===
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# ===== t-SNE Visualization =====
def plot_tsne_all_clusters(
    X, cluster_tree, gsrn_list=None, save_path="tsne_all_clusters.png"
):

    labels = np.full(len(X), -1)
    cluster_id = 0

    for node in cluster_tree.values():
        if "data_idx" in node and len(node["children"]) == 0:
            idx = node["data_idx"]
            labels[idx] = cluster_id
            cluster_id += 1

    print(" labeled samples:", np.sum(labels != -1))

    # t-SNE dimensionality reduction
    perplexity = min(30, max(5, len(X) // 3))
    tsne = TSNE(n_components=2, random_state=SEED, perplexity=perplexity)
    X_tsne = tsne.fit_transform(X)

    # display last 4 digits of GSRN if provided
    if gsrn_list is None:
        gsrn_list = list(map(str, range(len(X))))
    else:
        gsrn_list = [str(g)[-4:] for g in gsrn_list]

    # Plotting
    plt.figure(figsize=(12, 10))
    cmap = get_cmap("tab10")

    for i in np.unique(labels[labels != -1]):
        cluster_indices = np.where(labels == i)[0]
        plt.scatter(
            X_tsne[cluster_indices, 0],
            X_tsne[cluster_indices, 1],
            c=[cmap(i % 20)],
            label=f"Cluster {i} ({len(cluster_indices)})",
            s=50,
            alpha=0.7,
            edgecolors="k",
            linewidths=0.5,
        )
        # Add GSRN ID
        for j in cluster_indices:
            plt.text(
                X_tsne[j, 0],
                X_tsne[j, 1],
                gsrn_list[j],  # GSRN displayed on the plot
                fontsize=6,
                ha="center",
                va="center",
                color="black",
            )

    plt.title("t-SNE Visualization of Final Leaf Clusters")
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.legend(title="Clusters", loc="best", fontsize=9)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()
    print(f"t-SNE plot saved as: {save_path}")


# ===== PCA 3D Visualization =====
def extract_leaf_labels(X, cluster_tree):
    labels = np.full(len(X), -1)
    cluster_id = 0
    for node in cluster_tree.values():
        if "data_idx" in node and len(node["children"]) == 0:
            idx = node["data_idx"]
            labels[idx] = cluster_id
            cluster_id += 1
    return labels


def plot_pca_3d(X, labels, gsrn_list=None, save_path="pca_3d_clusters.png"):
    pca = PCA(n_components=3)
    X_pca = pca.fit_transform(X)

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection="3d")
    cmap = get_cmap("tab10")

    #  自动转换 gsrn 为字符串
    if gsrn_list is None:
        gsrn_list = list(map(str, range(len(X))))
    else:
        gsrn_list = [str(g)[-4:] for g in gsrn_list]

    for i in np.unique(labels[labels != -1]):
        indices = np.where(labels == i)[0]
        ax.scatter(
            X_pca[indices, 0],
            X_pca[indices, 1],
            X_pca[indices, 2],
            label=f"Cluster {i} ({len(indices)})",
            c=[cmap(i % 10)],
            s=40,
            alpha=0.7,
            edgecolors="k",
            linewidths=0.5,
        )
        for j in indices:
            ax.text(
                X_pca[j, 0],
                X_pca[j, 1],
                X_pca[j, 2],
                gsrn_list[j],
                fontsize=6,
                color="black",
            )

    ax.set_title("PCA 3D Visualization of Final Leaf Clusters")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()
    print(f"PCA 3D plot saved as: {save_path}")
