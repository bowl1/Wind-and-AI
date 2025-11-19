import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import pandas as pd
from sklearn.metrics import silhouette_score

df = pd.read_csv("../auto_split/400 turbines/selected turbines/selected_400_turbines_filtered.csv")
features = df[["GSRN", "UTM_x", "UTM_y"]].dropna()

X = df[["UTM_x", "UTM_y"]].values

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