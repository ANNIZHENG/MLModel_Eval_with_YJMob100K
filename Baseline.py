import pandas as pd
import math
from collections import Counter
from scipy.spatial import cKDTree

# Load Data
df_test = pd.read_csv('train.csv')
input_size = math.floor(584 * 0.8)
output_size = math.ceil(584 * 0.2)

# Store modes for each user and time slot
user_modes = {}
accurate_count = 0
total_euclidean_distance = 0
total_trajectory = 0
for uid, uid_group in df_test.groupby('uid'):
    train_data = uid_group.iloc[:input_size]
    test_data = uid_group.iloc[input_size:(input_size+output_size)]

    # Calculating mode for each t in train_data
    modes = {}
    for t, group in train_data.groupby('t'):
        most_common = Counter(group[['x', 'y']].apply(tuple, axis=1)).most_common(1)
        modes[t] = most_common[0][0] if most_common else None

    # Storing the modes by user and time slot
    user_modes[uid] = modes

    # Make a K-D Tree for mode
    mode_times = list(modes.keys())
    mode_tree = cKDTree([[t] for t in mode_times])

    # Determine if mode exist for that time
    for _, row in test_data.iterrows():
        t = row['t']
        actual_location = (row['x'],row['y'])
        predicted_location = (-1,-1)
        if t in modes and modes[t]:
            predicted_location = modes[t]
        else:
            # Find the nearest t with a recorded mode using KD Tree
            _, idx = mode_tree.query([[t]], k=1)
            nearest_t = mode_times[idx[0]]
            predicted_location = modes[nearest_t]
        
        # Compare predictions and actual values
        euclidean_distance = math.sqrt((actual_location[0] - predicted_location[0]) ** 2 + (actual_location[1] - predicted_location[1]) ** 2)
        total_trajectory += 1

        # Record distance and
        total_euclidean_distance += euclidean_distance
        if (euclidean_distance <= (1+math.sqrt(2))):
            accurate_count += 1

print(f"Average Euclidean Distance {(euclidean_distance/total_trajectory):.4f}, Accuracy: {(accurate_count/total_trajectory):.4f}")