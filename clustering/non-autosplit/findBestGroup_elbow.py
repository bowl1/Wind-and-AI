from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import pandas as pd

# === data loading ===
features_df = pd.read_csv(
    'auto_split/400 turbines/selected feature statistics/features_in_group_stage_400_turbines.csv'
)
features = features_df[[
    'mean_power_scaled',
    'std_power_scaled',
    'cv_scaled',
    'zero_ratio',
    'ramp_mean_scaled',
    'ramp_std_scaled'
]]

# === Elbow: inertia for different k ===
inertias = []
k_range = range(2, 15)   # 你想看的 k 范围

for k in k_range:
    kmeans = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=100,
        max_iter=300,
    )
    kmeans.fit(features)
    inertias.append(kmeans.inertia_)   # 簇内平方和

# === plotting ===
plt.figure(figsize=(8, 5))
plt.plot(k_range, inertias, marker='o')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Inertia (Within-Cluster Sum of Squares)')
plt.title('Elbow Method for Optimal k')
plt.grid(True)
plt.tight_layout()
plt.savefig('elbow_kmeans.png')
plt.show()