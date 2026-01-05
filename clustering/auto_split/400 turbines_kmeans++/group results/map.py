import pandas as pd
import ast
import geopandas as gpd
from shapely.geometry import Point
import matplotlib.pyplot as plt
import contextily as ctx

# ---------- 1. 读取 cluster_turbine_ids.csv 并展开 GSRN ----------
cluster_df = pd.read_csv("cluster_turbine_ids.csv")  

rows = []
for _, row in cluster_df.iterrows():
    cluster_id = row["Cluster"]
    gsrn_list_raw = row["GSRN"]

    # GSRN 列是字符串形式的列表
    if isinstance(gsrn_list_raw, str):
        gsrn_list = ast.literal_eval(gsrn_list_raw)
    else:
        gsrn_list = [gsrn_list_raw]

    for g in gsrn_list:
        rows.append({"Cluster": cluster_id, "GSRN": str(g)})

cluster_long = pd.DataFrame(rows)

# 只看唯一 GSRN 数量（应该是 400）
unique_gsrns = cluster_long["GSRN"].nunique()
print("Unique GSRNs in CSV:", unique_gsrns)

# ---------- 2. 从 masterdatawind 里拿坐标，并按 GSRN 去重 ----------
master = pd.read_parquet("../../../../energinet/masterdatawind.parquet")
master["GSRN"] = master["GSRN"].astype(str)

cols_keep = ["GSRN", "UTM_x", "UTM_y", "Capacity_kw", "Turbine_type", "In_service"]
master = master[cols_keep]

# 只保留 CSV 中的 GSRN
gsrn_set = set(cluster_long["GSRN"])
master_sub = master[master["GSRN"].isin(gsrn_set)].copy()

# 有些 GSRN 在 master 中可能有多行，这里按 In_service 排序后，每个 GSRN 只保留一行
master_sub = (
    master_sub
    .sort_values(["GSRN", "In_service"])
    .drop_duplicates(subset="GSRN", keep="last")
)

# 去掉没有坐标的
master_sub = master_sub.dropna(subset=["UTM_x", "UTM_y"])

print("Unique GSRNs in master_sub:", master_sub["GSRN"].nunique())

# ---------- 3. 合并 cluster + 坐标，只保留 400 台 ----------
merged = cluster_long.merge(master_sub, on="GSRN", how="inner")

print("Rows in merged:", len(merged))
print("Unique GSRNs in merged:", merged["GSRN"].nunique())

# ---------- 4. 转为 GeoDataFrame ----------
gdf = gpd.GeoDataFrame(
    merged,
    geometry=gpd.points_from_xy(merged["UTM_x"], merged["UTM_y"]),
    crs="EPSG:25832"   # UTM32
)
gdf_web = gdf.to_crs(epsg=3857)

# ---------- 5. 统计每个簇多少台风机 ----------
cluster_counts = gdf_web["Cluster"].value_counts().sort_index()
print("Turbine count per cluster:")
print(cluster_counts)

# ---------- 6. 画 Non-autosplit Turbine Clustering ----------
fig, ax = plt.subplots(figsize=(10, 10))

for cluster_id, group in gdf_web.groupby("Cluster"):
    n = len(group)
    group.plot(
        ax=ax,
        markersize=25,
        alpha=0.8,
        edgecolor="black",
        label = f"Cluster {cluster_id} ({n} turbines)"
    )

ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik)

ax.set_title("kmeans++ init_autoSplit_turbine_clustering", fontsize=14)
ax.set_axis_off()
ax.legend(title="Clusters", loc="lower left")

plt.tight_layout()
plt.show()
fig.savefig("kmeans++ init_autoSplit_turbine_clustering.png", dpi=300)