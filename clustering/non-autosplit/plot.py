import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# === Reproducibility ===
SEED = 42
random.seed(SEED)
np.random.seed(SEED)


def plot_tsne_and_pca_baseline(
    X,
    labels,
    gsrn_list=None,
    tsne_path="single_shot_tsne_clusters.png",
    pca3d_path="single_shot_pca_3d_clusters.png",
    title_prefix="Single-shot KMeans++ + FL",
):
    """
    X:        (N, d) 特征矩阵
    labels:   (N,)   聚类标签
    gsrn_list: 长度 N 的 GSRN 列表，用于显示后四位（可选）
    """

    labels = np.array(labels)
    valid_mask = labels != -1
    X_valid = X[valid_mask]
    labels_valid = labels[valid_mask]

    if gsrn_list is None:
        gsrn_valid = list(map(str, range(len(X_valid))))
    else:
        gsrn_valid = [str(g)[-4:] for i, g in enumerate(gsrn_list) if valid_mask[i]]

    print("有效样本数量:", np.sum(valid_mask))

    # =========================
    # 1) t-SNE 2D 可视化
    # =========================
    perplexity = min(30, max(5, len(X_valid) // 3))
    tsne = TSNE(n_components=2, random_state=SEED, perplexity=perplexity)
    X_tsne = tsne.fit_transform(X_valid)

    plt.figure(figsize=(12, 10))
    cmap = get_cmap("tab10")
    unique_labels = np.unique(labels_valid)

    for i in unique_labels:
        idx = np.where(labels_valid == i)[0]
        plt.scatter(
            X_tsne[idx, 0],
            X_tsne[idx, 1],
            c=[cmap(i % 20)],
            label=f"Cluster {i} ({len(idx)})",
            s=50,
            alpha=0.7,
            edgecolors="k",
            linewidths=0.5,
        )
        # 标注 GSRN 后四位
        for j in idx:
            plt.text(
                X_tsne[j, 0],
                X_tsne[j, 1],
                gsrn_valid[j],
                fontsize=6,
                ha="center",
                va="center",
                color="black",
            )

    plt.title(f"{title_prefix} – t-SNE 2D Visualization")
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.legend(title="Clusters", loc="best", fontsize=9)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(tsne_path, dpi=300)
    plt.show()
    print(f"t-SNE plot saved as: {tsne_path}")

    # =========================
    # 2) PCA 3D 可视化
    # =========================
    pca = PCA(n_components=3, random_state=SEED)
    X_pca = pca.fit_transform(X_valid)

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection="3d")
    cmap = get_cmap("tab10")

    for i in unique_labels:
        idx = np.where(labels_valid == i)[0]
        ax.scatter(
            X_pca[idx, 0],
            X_pca[idx, 1],
            X_pca[idx, 2],
            label=f"Cluster {i} ({len(idx)})",
            c=[cmap(i % 10)],
            s=40,
            alpha=0.7,
            edgecolors="k",
            linewidths=0.5,
        )
        for j in idx:
            ax.text(
                X_pca[j, 0],
                X_pca[j, 1],
                X_pca[j, 2],
                gsrn_valid[j],
                fontsize=6,
                color="black",
            )

    ax.set_title(f"{title_prefix} – PCA 3D Visualization")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")
    ax.legend()
    plt.tight_layout()
    plt.savefig(pca3d_path, dpi=300)
    plt.show()
    print(f"PCA 3D plot saved as: {pca3d_path}")