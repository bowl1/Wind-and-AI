from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import pandas as pd

# features_df = pd.read_csv('features_in_group_stage_50_turbines.csv')  
features_df = pd.read_csv('15 turbines/selected feature statistics/features_in_group_stage.csv')

features = features_df[[
    'mean_power_scaled',
    'std_power_scaled',
    'cv_scaled',
    'zero_ratio',
    'ramp_mean_scaled',
    'ramp_std_scaled'
]]

sil_scores = []
k_range = range(2, 11)  # try k=2-10

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=0)
    labels = kmeans.fit_predict(features)
    score = silhouette_score(features, labels)
    sil_scores.append(score)

# draw
plt.figure(figsize=(6, 4))
plt.plot(k_range, sil_scores, marker='o')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Score for Different k')
plt.grid(True)
plt.show()

