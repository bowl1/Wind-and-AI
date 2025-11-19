import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import contextily as ctx
from sklearn.cluster import KMeans
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D

# === Step 1: Load turbine coordinates ===
df = pd.read_csv("../auto_split/400 turbines/selected turbines/selected_400_turbines_filtered.csv")
df = df[["GSRN", "UTM_x", "UTM_y"]].dropna()

# === Step 2: Apply KMeans clustering on UTM coordinates ===
n_clusters = 7
kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
df["Cluster"] = kmeans.fit_predict(df[["UTM_x", "UTM_y"]])

# === Step 3: Convert to GeoDataFrame ===
gdf = gpd.GeoDataFrame(
    df,
    geometry=gpd.points_from_xy(df["UTM_x"], df["UTM_y"]),
    crs="EPSG:25832"  # UTM zone 32N (Europe, including Denmark)
).to_crs(epsg=3857)  # Convert to Web Mercator for contextily

# === Step 4: Assign colors per cluster ===
colors = cm.get_cmap('tab10', n_clusters)
cluster_ids = sorted(gdf["Cluster"].unique())
cluster_color_map = {c: mcolors.to_hex(colors(i)) for i, c in enumerate(cluster_ids)}

# === Step 5: Plotting ===
fig, ax = plt.subplots(figsize=(14, 14))
legend_elements = []

for cluster_id, group in gdf.groupby("Cluster"):
    group.plot(
        ax=ax,
        color=cluster_color_map[cluster_id],
        markersize=30,
        label=f"Cluster {cluster_id} ({len(group)} turbines)",
        alpha=0.8,
        edgecolor="black"
    )
    legend_elements.append(
        Line2D(
            [0], [0],
            marker='o',
            color='w',
            label=f"Cluster {cluster_id} ({len(group)} turbines)",
            markerfacecolor=cluster_color_map[cluster_id],
            markeredgecolor='black',
            markersize=8
        )
    )

ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik, zoom=11)
ax.set_title(" Wind Turbines Grouped by KMeans (7 Clusters)", fontsize=16)
ax.axis("off")
ax.legend(handles=legend_elements, loc="lower left", title="Clusters", fontsize=9, title_fontsize=11, frameon=True)

plt.savefig("wind_turbine_kmeans_map.png", dpi=300, bbox_inches='tight', pad_inches=0)

plt.tight_layout()
plt.show()

# === Step 6: Save cluster assignments to CSV ===
df[["GSRN", "Cluster"]].to_csv("kmeans_cluster_turbines.csv", index=False)