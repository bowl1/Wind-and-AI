from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import pandas as pd

# === data loading ===
features_df = pd.read_csv('auto_split/200 turbines/selected feature statistics/features_in_group_stage_200_turbines.csv')
features = features_df[[
    'mean_power_scaled',
    'std_power_scaled',
    'cv_scaled',
    'zero_ratio',
    'ramp_mean_scaled',
    'ramp_std_scaled'
]]

# === silhouette scores ===
sil_scores = []
k_range = range(2, 11)  # k from 2 to 10

# n_init = 10 → the algorithm initializes 10 times and keeps the best result
# max_iter = 300 → the algorithm performs up to 300 iterations per run

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=100)
    labels = kmeans.fit_predict(features)
    score = silhouette_score(features, labels)
    sil_scores.append(score)

# === plotting ===
plt.figure(figsize=(6, 4))
plt.plot(k_range, sil_scores, marker='o', label='KMeans')

for i, (k, score) in enumerate(zip(k_range, sil_scores)):
    plt.text(k, score + 0.01, f"{score:.4f}", ha='center', fontsize=9)

plt.xlabel('Number of clusters (k)')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Score for Different k')
plt.grid(True)
plt.tight_layout()
plt.savefig('silhouette_scores_kmeans.png')
plt.show()
