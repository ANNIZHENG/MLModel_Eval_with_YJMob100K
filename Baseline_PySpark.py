from pyspark.sql import SparkSession
from pyspark.sql.functions import col, expr, collect_list, size
from pyspark.sql.types import StructType, StructField, IntegerType, DoubleType
import math
from collections import Counter
from scipy.spatial import cKDTree

# Initialize Spark session
spark = SparkSession.builder \
    .appName("Baseline") \
    .getOrCreate()

# Define schema for CSV
schema = StructType([
    StructField("uid", IntegerType(), True),
    StructField("t", IntegerType(), True),
    StructField("x", DoubleType(), True),
    StructField("y", DoubleType(), True)
])

# Load Data
df_test = spark.read.csv('train.csv', header=True, schema=schema)

input_size = math.floor(584 * 0.8)
output_size = math.ceil(584 * 0.2)

# Register the DataFrame as a SQL temporary view
df_test.createOrReplaceTempView("user_data")

# Store modes for each user and time slot
user_modes = {}
accurate_count = 0
total_euclidean_distance = 0
total_trajectory = 0

# Get unique uids
uids = df_test.select("uid").distinct().collect()

for uid_row in uids:
    uid = uid_row.uid
    uid_group = spark.sql(f"SELECT * FROM user_data WHERE uid = {uid}")

    train_data = uid_group.limit(input_size)
    test_data = uid_group.subtract(train_data).limit(output_size)

    # Calculating mode for each t in train_data
    modes = {}
    train_data_grouped = train_data.groupBy("t").agg(collect_list("x").alias("x_list"), collect_list("y").alias("y_list"))
    for row in train_data_grouped.collect():
        t = row.t
        most_common = Counter(list(zip(row.x_list, row.y_list))).most_common(1)
        modes[t] = most_common[0][0] if most_common else None

    # Storing the modes by user and time slot
    user_modes[uid] = modes

    # Make a K-D Tree for mode
    mode_times = list(modes.keys())
    mode_tree = cKDTree([[t] for t in mode_times])

    # Determine if mode exists for that time
    for row in test_data.collect():
        t = row.t
        actual_location = (row.x, row.y)
        predicted_location = (-1, -1)
        if t in modes and modes[t]:
            predicted_location = modes[t]
        else:
            # Find the nearest t with a recorded mode using KD Tree
            _, idx = mode_tree.query([[t]], k=1)
            nearest_t = mode_times[idx[0][0]]
            predicted_location = modes[nearest_t]

        # Compare predictions and actual values
        euclidean_distance = math.sqrt((actual_location[0] - predicted_location[0]) ** 2 + (actual_location[1] - predicted_location[1]) ** 2)
        total_trajectory += 1

        # Record distance and accuracy
        total_euclidean_distance += euclidean_distance
        if euclidean_distance <= (1 + math.sqrt(2)):
            accurate_count += 1

# Output results
average_euclidean_distance = total_euclidean_distance / total_trajectory
accuracy = accurate_count / total_trajectory
print(f"Average Euclidean Distance: {average_euclidean_distance:.4f}, Accuracy: {accuracy:.4f}")

# Stop the Spark session
spark.stop()
