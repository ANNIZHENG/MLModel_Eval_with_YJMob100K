import pandas as pd
from collections import Counter
from scipy.spatial import cKDTree

# Load Data
df_test = pd.read_csv('test.csv')

# Store modes for each user and time slot
user_modes = {}

# Placeholder to collect predictions and actual values for accuracy calculation
predictions = []
actuals = []

for uid, uid_group in df_test.groupby('uid'):
    train_data = uid_group.iloc[:48]
    test_data = uid_group.iloc[48:(48+48)]

    # Calculating mode for each t in train_data
    modes = {}
    for t, group in train_data.groupby('t'):
        most_common = Counter(group[['x', 'y']].apply(tuple, axis=1)).most_common(1)
        modes[t] = most_common[0][0] if most_common else None

    # Storing the modes by user and time slot
    user_modes[uid] = modes

    # KD-tree to find nearest t with mode available
    if modes:
        mode_times = list(modes.keys())
        mode_tree = cKDTree([[t] for t in mode_times])

    # Predicting locations in test_data based on the computed modes
    for _, row in test_data.iterrows():
        t = row['t']
        if t in modes and modes[t]:
            predicted_location = modes[t]
        else:
            # Find the nearest t with a recorded mode
            _, idx = mode_tree.query([[t]], k=1)
            nearest_t = mode_times[idx[0]]
            predicted_location = modes[nearest_t]

        # Collect predictions and actual values
        predictions.append(predicted_location)
        actuals.append((row['x'], row['y']))

# Accuracy
accuracy = sum(1 for pred, act in zip(predictions, actuals) if pred == act) / len(predictions)
print(f"Accuracy: {accuracy}")