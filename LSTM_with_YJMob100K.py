import pandas as pd
import numpy as np
import math
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader, Sampler
from torch.nn.utils.rnn import pad_sequence

# Load data with users from yjmob1
df_train = pd.read_csv('train.csv')
df_test  = pd.read_csv('test.csv')
df_true_test = pd.read_csv('true_test.csv')

# Adjust input and output size here
input_size  = 192
output_size = 48

# Prepare Dataset as Model Input
class TrajectoryDataset(Dataset):
    def __init__(self, all_data, input_size, output_size):
        self.data = []

        # Window would partition one Trajectory
        window_size = input_size + output_size

        # The Dataset would store one Trajectory data
        # Trajectory: 48*4 input data, 48*4 corresponding time-step, 48 output data, 48 corresponding time-step
        for i in range(0, len(all_data)-window_size+1, window_size):
            uid = all_data.iloc[i]['uid']
            xy = all_data.iloc[i:i+window_size]['combined_xy'].tolist()
            t = all_data.iloc[i:i+window_size]['t'].tolist()
            self.data.append((uid, xy[:input_size], xy[input_size:], t[:input_size], t[input_size:]))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        user_id, inputs, labels, positions, label_positions = self.data[idx]
        return torch.tensor(user_id), torch.tensor(inputs), torch.tensor(labels), torch.tensor(positions), torch.tensor(label_positions)

# Create Dataset
train_dataset = TrajectoryDataset(df_train, input_size, output_size)
test_dataset = TrajectoryDataset(df_test, input_size, output_size)

# The Sampler is used to Aggregate Trajectory of the Same User into the Same Batch
class UserGroupSampler(Sampler):
    def __init__(self, dataset):
        self.indices_by_user = {} # use dictionary to store info
        for idx in range(len(dataset)):
            uid, _, _, _, _ = dataset[idx]
            uid = uid.item()
            if uid not in self.indices_by_user:
                self.indices_by_user[uid] = []
            self.indices_by_user[uid].append(idx)

    def __iter__(self):
        for _, indices in self.indices_by_user.items():
            yield indices

    def __len__(self):
        return len(self.indices_by_user)

# Aggregation Function for Making Sure All Trajectories are in the Same Length in One Batch
def collate_fn(batch):
    user_ids, inputs_batch, labels_batch, positions_batch, label_positions_batch = zip(*batch)
    inputs_padded = pad_sequence(inputs_batch, batch_first=True, padding_value=0)
    labels_padded = pad_sequence(labels_batch, batch_first=True, padding_value=0)
    positions_padded = pad_sequence(positions_batch, batch_first=True, padding_value=0)
    label_positions_padded = pad_sequence(label_positions_batch, batch_first=True, padding_value=0)
    return user_ids[0].clone().detach(), inputs_padded, labels_padded, positions_padded, label_positions_padded

# Create Sampler and DataLoader
train_sampler = UserGroupSampler(train_dataset) # group data of the same user id on the same batch
train_dataloader = DataLoader(train_dataset, batch_sampler=train_sampler, collate_fn=collate_fn)
test_sampler = UserGroupSampler(test_dataset)
test_dataloader = DataLoader(test_dataset, batch_sampler=test_sampler, collate_fn=collate_fn)

print("Training data and Testing data loaded!")

# Load Built-In LSTM Model
class LSTMModel(nn.Module):
    def __init__(self, loc_size, output_size, embed_dim, hidden_size, num_layers, device):
        super(LSTMModel, self).__init__()
        self.input_embedding = nn.Embedding(loc_size, embed_dim)
        self.lstm = nn.LSTM(input_size=embed_dim, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, loc_size)
        self.output_size = output_size
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.device = device

    def forward(self, x):
        x = self.input_embedding(x)  # Positional embedding
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device) # Initialize hidden state
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device) # Initialize cell state
        
        out, _ = self.lstm(x, (h0, c0))  # LSTM output
        out = out[:, -self.output_size:, :]  # Take the last `output_size` time steps
        out = self.fc(out)  # Apply the final linear layer to each time step

        return out

# Instantiate the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LSTMModel(loc_size=40000, output_size=output_size, embed_dim=512, hidden_size=256, num_layers=1, device=device)
model.to(device)

print("PyTorch Built-in LSTM loaded!")

# Function used to decode grid-id to (x,y) format
def decode_token_to_cell(token_id, num_cells_per_row):
    cell_x = token_id % num_cells_per_row + 1
    cell_y = token_id // num_cells_per_row + 1 
    return cell_x, cell_y

# Function used to decode the whole trajectory (a sequence of grid-ids)
def decode_trajectory(traj, num_cells_per_row=200):
    decoded_traj = [decode_token_to_cell(token_id, num_cells_per_row) for token_id in traj]
    return decoded_traj

# Model Training Set-up
def train(model, dataloader, device, learning_rate, threshold=(1+math.sqrt(2))):
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=learning_rate)
    
    total_loss = 0.0
    total_distance = 0.0
    total_trajectories = 0
    correct_trajectories = 0
    
    for _, inputs, labels, _, _ in dataloader: 
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs.view(-1, outputs.size(-1)), labels.view(-1))
        
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # Get the index of the max log-probability
        _, predicted = outputs.max(2)

        # Calculate accuracy for each location
        for i in range(labels.size(0)):
            true_traj = labels[i].cpu().numpy()
            pred_traj = predicted[i].cpu().numpy()

            decoded_true_traj = np.array(decode_trajectory(true_traj))
            decoded_pred_traj = np.array(decode_trajectory(pred_traj))

            # Calculate Euclidean distances for each point
            euclidean_distances = np.linalg.norm(decoded_true_traj - decoded_pred_traj, axis=1)
            total_distance += np.sum(euclidean_distances)
            total_trajectories += len(euclidean_distances)

            # Increment correct count for distances below the threshold
            correct_trajectories += np.sum(euclidean_distances < threshold)
    
    avg_loss = total_loss / len(dataloader)
    avg_euclidean_distance = total_distance / total_trajectories
    accuracy = correct_trajectories / total_trajectories

    print(f"Average Euclidean Distance Difference: {avg_euclidean_distance:.4f}, Accuracy: {accuracy:.4f}")
    
    return avg_loss, avg_euclidean_distance, accuracy

# Training Set-up (with Epochs)
def train_model(model, dataloader, device, epochs, learning_rate):
    for epoch in range(epochs):
        print(f"Train: Epoch {epoch+1}")
        train(model, dataloader, device, learning_rate)

print("Start Training Process!")

train_model(model=model, dataloader=train_dataloader, device=device, epochs=5, learning_rate=0.001)

# Exapnd prediction to prepare to correspond to ground truth
def expand_predictions(predicted_locs, predicted_times, max_time=47):
    expanded_locs = []
    expanded_times = list(range(max_time + 1))
    current_loc = predicted_locs[0]
    loc_dict = dict(zip(predicted_times, predicted_locs))
    for time in expanded_times: # for time in the 48 time indices
        if time in loc_dict: # if time is predicted
            # then use the prediction
            current_loc = loc_dict[time] # then use the prediction
        else: # if time is not predicted
            # then use the previous/later prediction
            found = False
            if time > 0: # search the previous predictions
                for j in range(time-1,-1,-1):
                    prev_time = predicted_times[j]
                    if prev_time in loc_dict:
                        current_loc = loc_dict[prev_time]
                        found = True
                        break
            if not found: # search the later predictions
                for j in range(time+1, len(predicted_times)):
                    next_time = predicted_times[j]
                    if next_time in loc_dict:
                        current_loc = loc_dict[next_time]
                        found = True
                        break
            if not found: # last check
                current_loc = predicted_locs[0]
        expanded_locs.append(current_loc)
    return expanded_locs, expanded_times

# Measure whether the predicted location falls within threshold
def accuracy_measure(user_id, predicted_locs, predicted_times, true_locs, true_times):
    expanded_locs, expanded_times = expand_predictions(predicted_locs, predicted_times)
    matched_locs = []

    # Select only the prediction with a corresponding ground truth
    for true_time in true_times:
        if true_time in expanded_times:
            index = expanded_times.index(true_time)
            matched_locs.append(expanded_locs[index])
        else:
            matched_locs.append(None)

    # Convert prediction to (x,y) trajectory
    matched_locs = np.array(decode_trajectory(matched_locs))

    # Inference
    threshold = 1+math.sqrt(2)
    total_distance = 0.0
    total_location = 0
    correct_location = 0

    total_distance_nextplace = 0.0
    total_location_nextplace = 0
    correct_location_nextplace = 0

    # Accuracy measure based on Euclidean Distance difference
    # Next Place Prediction
    matched_locs_nextplace = matched_locs[0]
    euclidean_distance_nextplace = math.sqrt((true_locs[0][0] - matched_locs_nextplace[0])**2 + (true_locs[0][1] - matched_locs_nextplace[1])**2)
    total_distance_nextplace += euclidean_distance_nextplace
    total_location_nextplace += 1
    if (euclidean_distance_nextplace < threshold):
        correct_location_nextplace += 1

    # Next Sequence Prediction
    euclidean_distances = np.linalg.norm(true_locs - matched_locs, axis=1)
    total_distance += np.sum(euclidean_distances)

    for euclidean_distance in euclidean_distances:
        total_location += 1
        if (euclidean_distance < threshold):
            correct_location += 1
    
    # avg_euclidean_distance = total_distance / total_location 
    # accuracy = correct_location / total_location
    # print(f"User ID: {user_id}, Average Euclidean Distance Difference: {avg_euclidean_distance:.4f}, Accuracy: {accuracy:.4f}")

    return matched_locs, total_distance, total_location, correct_location, matched_locs_nextplace, total_distance_nextplace, total_location_nextplace, correct_location_nextplace

# Inference part
# Currentl doing Prediction assuming user would follow the predicted 48 step trajectory throughout 15 days
# You may comment out the Autoregressive part, if the above way is not preferred
def recursive_inference_per_user(model, dataloader, device, true_data):
    # Measurement used for total accuracy calculation
    total_distances = 0.0 # total distance off
    total_locations = 0 # total num of location to be predicted
    correct_locations = 0 # total num of locations that are correctly predicted

    total_distances_nextplace = 0.0
    total_locations_nextplace = 0
    correct_locations_nextplace = 0

    model.eval()
    
    predictions = {} # Store the 15-day predicted values per user
    predictions_time = {} # Store the corresponding 15-day time values per user

    predictions_nextplace = {} 
    
    with torch.no_grad():
        for user_id, inputs, _, _, label_positions in dataloader: 

            user_id = user_id.item()
            inputs = inputs.to(device) # the location
            label_positions = label_positions.to(device) # the time

            # Ground Truth Data import
            true_data_by_uid = true_data[true_data['uid']==user_id]
            true_locs = np.array(list(zip(true_data_by_uid['x'], true_data_by_uid['y']))) # ground truth location
            true_times = true_data_by_uid['t'].to_list() # ground truth time
            num_predictions = len(true_data_by_uid) # number of needed prediction
            
            # Store predictions and times for the current user
            user_predictions = []
            user_predictions_time = []

            # Initial Prediction
            outputs = model(inputs) # Get the initial prediction on location
            _, predicted = outputs.max(2)  # Get the index of the max log-probability

            for i in range(inputs.size(0)):
                user_predictions.extend(predicted[i].cpu().numpy())
                user_predictions_time.extend(label_positions[i].cpu().numpy())
            
            '''
            # The Autoregressive Prediction witout assuming the predicted 48-time-step is uniform throughout the user movement
            # Set up for Autoregressive test
            current_input = inputs # the input locations
            current_label = predicted # the predicted locations
            current_label_positions = label_positions # the predicted times
            
            for i in range(current_input.size(0)):
                user_predictions.extend(predicted[i].cpu().numpy()) # store the predicted locations
                user_predictions_time.extend(label_positions[i].cpu().numpy()) # store the predicted times

            # Autoregressive Prediction
            while (len(user_predictions) < num_predictions):
                # Initialize batch-related variables for prediction
                new_inputs = []
                new_label_positions = []
                
                # Create New Batch as Input
                for i in range(current_input.size(0)):
                    # New Input = (192-48) Past Location + 48 Predicted Location
                    true_traj = current_input[i].cpu().numpy() # len: 192 # the input locations
                    pred_traj = current_label[i].cpu().numpy() # len: 48  # the predicted locations

                    # Truncate and Concatenate to create new input
                    new_input_traj = torch.tensor(np.concatenate((true_traj[48:], pred_traj))).to(device) # Concatenated prediction (the new input)
                    new_label_position = current_label_positions[i].to(device) # the predicted time (the known time)

                    new_inputs.append(new_input_traj) # Append concatenated prediction Location, i.e., the new input
                    new_label_positions.append(new_label_position) # Append the predicted Time
                
                # Stack info to form a New Batch
                current_input = torch.stack(new_inputs)
                current_label_positions = torch.stack(new_label_positions)
                
                # Create New Prediction
                new_outputs = model(current_input) 
                _, predicted = new_outputs.max(2)
                
                for i in range(current_input.size(0)):
                    user_predictions.extend(predicted[i].cpu().numpy())
                    user_predictions_time.extend(current_label_positions[i].cpu().numpy())

                current_label = predicted
            '''
            
            # Measure accuracy for each user's prediction
            matched_locs, total_distance, total_location, correct_location, matched_locs_nextplace, total_distance_nextplace, total_location_nextplace, correct_location_nextplace = accuracy_measure(user_id, user_predictions, user_predictions_time, true_locs, true_times)

            # Store predictions and times for the user
            predictions[user_id] = matched_locs
            predictions_time[user_id] = true_times

            predictions_nextplace[user_id] = matched_locs_nextplace # next place prediction

            # Record the total distance
            total_distances += total_distance
            total_locations += total_location
            correct_locations += correct_location

            total_distances_nextplace += total_distance_nextplace # next place prediction
            total_locations_nextplace+= total_location_nextplace # next place prediction
            correct_locations_nextplace += correct_location_nextplace # next place prediction

    avg_euclidean_distance = total_distances / total_locations
    accuracy = correct_locations / total_locations

    avg_euclidean_distance_nextplace = total_distances_nextplace / total_locations_nextplace
    accuracy_nextplace = correct_locations_nextplace / total_locations_nextplace

    print(f"Total Users' Adjusted Euclidean Distance Difference: {avg_euclidean_distance:.4f}, Accuracy: {accuracy:.4f}")
    print(f"Total Users' Adjusted Euclidean Distance Difference of Next-Place Prediction: {avg_euclidean_distance_nextplace:.4f}, Accuracy: {accuracy_nextplace:.4f}")

    return avg_euclidean_distance, accuracy, predictions, predictions_time, predictions_nextplace

# Inference
print("Test")
_, _, predictions, predictions_time, predictions_nextplace = recursive_inference_per_user(model, test_dataloader, device, df_true_test)

# Output data to csv
import csv

# Output predicted trajectory data as CSV
csv_data = []
for uid in predictions:
    locations = predictions[uid].tolist()
    times = predictions_time[uid]
    for time, location in zip(times, locations):
        location.extend([time, uid])
        csv_data.append(location)

# Write data to CSV file
with open('lstm_prediction.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['x', 'y', 't', 'uid']) 
    writer.writerows(csv_data)

'''
# Output next-place predicted trajectory data as CSV
csv_data_nextplace = []
for uid in predictions_nextplace:
    location = predictions_nextplace[uid].tolist()
    time = predictions_time[uid][0]
    location.extend([time, uid])
    csv_data_nextplace.append(location)

# Write data to CSV file
with open('lstm_prediction_nextplace.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['x', 'y', 't', 'uid']) 
    writer.writerows(csv_data_nextplace)
'''

print("Predicted trajectories written to the csv file")