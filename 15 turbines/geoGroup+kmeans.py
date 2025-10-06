import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ==== parameters ====
n_geo_groups = 5       # geo groups
random_state = 42

# ==== load data ====
base_dir = os.path.dirname(os.path.abspath(__file__))
df_feat = pd.read_csv(os.path.join(base_dir, "selected_feature_statistics/features_in_group_stage.csv"))
df_geo = pd.read_csv(os.path.join(base_dir, "selected_turbines/selected_15_turbines_filtered.csv"))[["GSRN", "UTM_x", "UTM_y"]]
df = pd.merge(df_feat, df_geo, on="GSRN")

# ==== feature columns ====
feature_cols = [
    "mean_power_scaled",
    "std_power_scaled",
    "cv_scaled",
    "zero_ratio",
    "ramp_mean_scaled",
    "ramp_std_scaled"
]
X = df[feature_cols].values

# ==== geo clustering ====
geo_kmeans = KMeans(n_clusters=n_geo_groups, random_state=random_state)
df["geo_group"] = geo_kmeans.fit_predict(df[["UTM_x", "UTM_y"]])

# ==== calculate silhouette score based on geo_group labels ====
sil_score_geo = silhouette_score(X, df["geo_group"])
print(f"\nSilhouette Score for Geo-based Clustering (based on features): {sil_score_geo:.4f}")

# ==== PCA Visualization for geo_group ====
X_pca = PCA(n_components=2).fit_transform(X)
plt.figure(figsize=(10, 7))
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=df["geo_group"], palette="Set2", s=100)

for i, gsrn_id in enumerate(df["GSRN"]):
    plt.text(X_pca[i, 0]+0.03, X_pca[i, 1]+0.03, str(gsrn_id)[-4:], fontsize=8)

plt.title(f"PCA of Geo Grouping (n={n_geo_groups}, Silhouette={sil_score_geo:.2f})")
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.grid(True)
plt.tight_layout()
plt.savefig(f"geo_grouping_pca_n{n_geo_groups}.png", dpi=300)
plt.show()