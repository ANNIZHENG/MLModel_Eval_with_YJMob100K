import pandas as pd
import numpy as np

# Function used to decode grid-id to (x,y) format
def decode_token_to_cell(token_id, num_cells_per_row):
    cell_x = token_id % num_cells_per_row + 1
    cell_y = token_id // num_cells_per_row + 1 
    return cell_x, cell_y

# Load Data
df_test = pd.read_csv('test.csv')
df_true_test = pd.read_csv('true_test.csv')

# Store modes for each user and time slot
all_predictions = {}
all_prediction_times = {}
accurate_count = 0
total_euclidean_distance = 0
total_location = 0

# Baseline Model
# For each user data
for uid, train_data in df_test.groupby('uid'):
    test_data = df_true_test[df_true_test['uid']==uid]

    # Initialize placement for storing locations
    mode = {}
    predictions = []
    prediction_times = []
    time_recorded_for_mode = set()

    # Calculate Frequency
    for t, group in train_data.groupby('t'):
        x,y = decode_token_to_cell(group.value_counts(['combined_xy']).index[0][0],200)
        mode[t] = (x,y)
        time_recorded_for_mode.add(t)

    # Populate frequency records (i.e., predicitons)
    for i in range(len(test_data)):
        total_location += 1
        test_data_row = test_data.iloc[i]
        time = test_data_row['t']
        prediction = mode.get(time)
        x1,y1 = -1,-1

        if prediction is None:
            found = False
            if i > 0: # Search Previous
                for j in range(i-1, -1, -1):
                    prev_time = test_data.iloc[j]['t']
                    if prev_time in mode:
                        x1, y1 = mode[prev_time] # Find Nearest Prediction
                        mode[time] = (x1,y1) # Store Nearest Prediction
                        predictions.append([x1,y1])
                        prediction_times.append(time)
                        found = True
                        break
            if not found: # Search Later
                for j in range(i+1, len(test_data)):
                    next_time = test_data.iloc[j]['t']
                    if next_time in mode:
                        x1,y1 = mode[next_time]
                        mode[time] = (x1,y1)
                        predictions.append([x1,y1])
                        prediction_times.append(time)
                        found = True
                        break
            if not found: # Last Check
                x1, y1 = 0,0
                predictions.append([x1,y1])
                prediction_times.append(time)
        else:
            x1,y1 = prediction
            predictions.append([x1,y1])
            prediction_times.append(time)

        x2,y2 = test_data_row['x'], test_data_row['y'] # Ground True
        euclidean_distance = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        if (euclidean_distance <= (1+np.sqrt(2))):
            accurate_count += 1
        total_euclidean_distance += euclidean_distance

    all_predictions[uid] = predictions
    all_prediction_times[uid] = prediction_times

print(f"Average Euclidean Distance {(total_euclidean_distance/total_location):.4f}, Accuracy: {(accurate_count/total_location):.4f}")

# Output data to csv
import csv

# Output predicted trajectory data as CSV
csv_data = []
for uid in all_predictions:
    locations = all_predictions[uid]
    times = all_prediction_times[uid]
    for time, location in zip(times, locations):
        location.extend([time, uid])
        csv_data.append(location)

# Write data to CSV file
with open('baseline_prediction.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['x', 'y', 't', 'uid']) 
    writer.writerows(csv_data)

print("Predicted trajectories written to the csv file")