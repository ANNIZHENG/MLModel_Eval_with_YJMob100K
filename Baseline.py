import pandas as pd
import numpy as np
import math

# Load Data
df_test = pd.read_csv('train.csv')
input_size = math.floor(584 * 0.8)
output_size = math.ceil(584 * 0.2)

# Store modes for each user and time slot
user_modes = {}
accurate_count = 0
total_euclidean_distance = 0
total_trajectory = 0

# Baseline Model
for uid, uid_group in df_test.groupby('uid'):
    total_trajectory += 1
    train_data = uid_group.iloc[:input_size]
    test_data = uid_group.iloc[input_size:(input_size+output_size)]

    mode = {}
    time_recorded_for_mode = set()

    for t, group in train_data.groupby('t'):
        x,y = group.value_counts(['x', 'y']).index[0]
        mode[t] = (x,y)
        time_recorded_for_mode.add(t)

    for i in range(len(test_data)):
        test_data_row = test_data.iloc[i]
        time = test_data_row['t']
        prediction = mode.get(time)
        x1,y1 = -1,-1

        if prediction is None:
            found = False
            if i > 0:
                for j in range(i-1, -1, -1):
                    prev_time = test_data.iloc[j]['t']
                    if prev_time in mode:
                        x1, y1 = mode[prev_time]
                        found = True
                        break
            if not found:
                for j in range(i+1, len(test_data)):
                    next_time = test_data.iloc[j]['t']
                    if next_time in mode:
                        x1, y1 = mode[next_time]
                        found = True
                        break
            if not found:
                x1, y1 = 0, 0
        else:
            x1, y1 = prediction

        x2,y2 = test_data_row['x'], test_data_row['y'] 
        euclidean_distance = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        if (euclidean_distance <= (1+np.sqrt(2))):
            accurate_count += 1
        total_euclidean_distance += euclidean_distance

print(f"Average Euclidean Distance {(total_euclidean_distance/total_trajectory):.4f}, Accuracy: {(accurate_count/total_trajectory):.4f}")