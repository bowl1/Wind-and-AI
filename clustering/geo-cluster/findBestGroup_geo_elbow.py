import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import pandas as pd

df = pd.read_csv("../auto_split/400 turbines/selected turbines/selected_400_turbines_filtered.csv")
df = df[["GSRN", "UTM_x", "UTM_y"]].dropna()

X = df[["UTM_x", "UTM_y"]].values

inertias = []
K_range = range(2, 15)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
    kmeans.fit(X)
    inertias.append(kmeans.inertia_)  # Inertia = sum of squared distances to cluster center

plt.figure(figsize=(8, 5))
plt.plot(K_range, inertias, marker='o')
plt.title("Elbow Method for Optimal k")
plt.xlabel("Number of clusters (k)")
plt.ylabel("Inertia (Within-Cluster Sum of Squares)")
plt.grid(True)
plt.savefig('elbow_kmeans.png')
plt.show()