import os
import pickle

from pyspark.sql import SparkSession
from pyspark.sql.window import Window
from pyspark.sql.functions import col, lag, row_number, count
from pyspark.ml.feature import VectorAssembler, MinMaxScaler
from pyspark.ml.functions import vector_to_array
from pyspark.ml import Pipeline
from sklearn.preprocessing import MinMaxScaler as SklearnMinMaxScaler
import numpy as np

spark = SparkSession.builder \
    .appName("turbine_preprocess_split") \
    .getOrCreate()

# =========================
# 1. READ CSV
# =========================

df = spark.read.csv(
    "../../dataset/turbine_prediction_selected_features_400turbines.csv",
    header=True,
    inferSchema=True
)

# =========================
# 2. COLUMN SETUP
# =========================

target_col = "power_output"
static_cols = ["Capacity_kw", "age"]
identifier_col = "GSRN"

dynamic_cols = [
    c for c in df.columns
    if c not in static_cols + [identifier_col, "timestamp", target_col]
]

feature_cols = dynamic_cols + static_cols

# =========================
# 3. LAG FEATURES
# =========================

window = Window.partitionBy(identifier_col).orderBy("timestamp")

num_lags = 24
lagged_cols = []

for col_name in dynamic_cols:
    for i in range(1, num_lags + 1):
        new_col = f"{col_name}_lag_{i}"
        df = df.withColumn(
            new_col,
            lag(col_name, i).over(window)
        )
        lagged_cols.append(new_col)

df = df.dropna()

print("Rows after lag/dropna:", df.count())

# =========================
# 4. TRAIN / VAL SPLIT
# =========================

# 每个 turbine 按时间编号
split_window = Window.partitionBy(identifier_col).orderBy("timestamp")

df = df.withColumn(
    "row_id",
    row_number().over(split_window)
)

length_df = df.groupBy(identifier_col).agg(
    count("*").alias("n")
)

df = df.join(length_df, on=identifier_col)

# 前 80% → train
train_df = df.filter(col("row_id") <= col("n") * 0.8)

# 后 20% → val
val_df = df.filter(col("row_id") > col("n") * 0.8)

print("Train rows:", train_df.count())
print("Val rows:", val_df.count())

# =========================
# 5. SCALING (FIT ON TRAIN ONLY)
# =========================

all_features = static_cols + lagged_cols

assembler = VectorAssembler(
    inputCols=all_features,
    outputCol="features_vec"
)

scaler = MinMaxScaler(
    inputCol="features_vec",
    outputCol="scaled_features"
)

pipeline = Pipeline(stages=[assembler, scaler])

# ⚠️ 只用 train fit
scaler_model = pipeline.fit(train_df)

train_df = scaler_model.transform(train_df)
val_df = scaler_model.transform(val_df)

scaled_array = vector_to_array(col("scaled_features"))

for i, name in enumerate(all_features):
    train_df = train_df.withColumn(name, scaled_array[i])
    val_df = val_df.withColumn(name, scaled_array[i])

# =========================
# 6. SCALE TARGET (FIT ON TRAIN)
# =========================

target_assembler = VectorAssembler(
    inputCols=[target_col],
    outputCol="target_vec"
)

target_scaler = MinMaxScaler(
    inputCol="target_vec",
    outputCol="scaled_target"
)

target_pipeline = Pipeline(stages=[target_assembler, target_scaler])

target_model = target_pipeline.fit(train_df)

train_df = target_model.transform(train_df)
val_df = target_model.transform(val_df)

scaled_target = vector_to_array(col("scaled_target"))

train_df = train_df.withColumn(target_col, scaled_target[0])
val_df = val_df.withColumn(target_col, scaled_target[0])

# =========================
# 7. FINAL COLUMNS
# =========================

final_cols = (
    [identifier_col, target_col]
    + static_cols
    + lagged_cols
)

train_df = train_df.select(*final_cols)
val_df = val_df.select(*final_cols)

# =========================
# 8. WRITE PARQUET
# =========================

train_df.write \
    .partitionBy(identifier_col) \
    .mode("overwrite") \
    .parquet("../processed/train")

val_df.write \
    .partitionBy(identifier_col) \
    .mode("overwrite") \
    .parquet("../processed/val")

# =========================
# 9. SAVE GLOBAL Y_SCALER
# =========================
# Reconstruct a sklearn-compatible MinMaxScaler from Spark's PipelineModel
# so the notebook can apply inverse_transform for comparable metrics.

spark_y_scaler = target_model.stages[1]  # the Spark MinMaxScaler stage

sk_y_scaler = SklearnMinMaxScaler()
data_min = spark_y_scaler.originalMin.toArray()
data_max = spark_y_scaler.originalMax.toArray()
data_range = data_max - data_min

sk_y_scaler.data_min_ = data_min
sk_y_scaler.data_max_ = data_max
sk_y_scaler.data_range_ = data_range
sk_y_scaler.scale_ = np.where(data_range == 0, 0.0, 1.0 / data_range)
sk_y_scaler.min_ = -data_min * sk_y_scaler.scale_
sk_y_scaler.feature_range = (0, 1)
sk_y_scaler.n_features_in_ = len(data_min)
sk_y_scaler.n_samples_seen_ = train_df.count()

scaler_save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "processed_data", "global_y_scaler.pkl")
os.makedirs(os.path.dirname(scaler_save_path), exist_ok=True)
with open(scaler_save_path, "wb") as f:
    pickle.dump(sk_y_scaler, f)

print(f"Saved global y_scaler → {scaler_save_path}")
print(f"  power_output original range: [{data_min[0]:.4f}, {data_max[0]:.4f}]")
print("Spark preprocessing with split complete!")