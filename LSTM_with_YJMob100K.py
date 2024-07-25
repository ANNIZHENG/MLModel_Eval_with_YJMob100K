import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
import math

# Load data with users from yjmob1
df_train = pd.read_csv('train.csv')
df_test  = pd.read_csv('test.csv')

# Group data by uid
grouped_data_train = [group for _, group in df_train.groupby('uid')]
grouped_data_test  = [group for _, group in df_test.groupby('uid')]

# Adjust input and output size here
input_size  = 48 * 2
output_size = 48

class TrajectoryDataset(Dataset):
    def __init__(self, grouped_data, input_size, output_size, test=False):
        self.data = []
        if (test):
            for group in grouped_data:
                uid = group['uid'].values.tolist()
                xy = group['combined_xy'].values.tolist()
                t = group['t'].values.tolist()

                for i in range(0, len(group), input_size):
                    input_end = i+input_size
                    # user_id, inputs, positions
                    self.data.append((uid[0], xy[i:input_end], [], t[i:input_end], []))
        else:
            for group in grouped_data:
                uid = group['uid'].values.tolist()
                xy = group['combined_xy'].values.tolist()
                t = group['t'].values.tolist()

                window_size = input_size+output_size
                for i in range(0, len(group)-window_size+1, window_size):
                    input_end = i + input_size
                    # user_id, inputs, labels, positions, label_positions
                    self.data.append((uid[0], xy[i:input_end], xy[input_end:(input_end+output_size)], t[i:input_end], t[input_end:(input_end+output_size)]))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        user_id, inputs, labels, positions, label_positions = self.data[idx]
        return torch.tensor(user_id), torch.tensor(inputs), torch.tensor(labels), torch.tensor(positions), torch.tensor(label_positions)

train_dataset = TrajectoryDataset(grouped_data_train, input_size, output_size, False)
test_dataset  = TrajectoryDataset(grouped_data_test,  input_size, output_size, True)

# Clutch train and test datasets into dataloaders
def collate_fn(batch):
    # Unzip all batch
    user_id, inputs_batch, labels_batch, positions_batch, label_positions_batch = zip(*batch)

    # Unpack user_id
    unpacked_user_id = []
    for uid in user_id:
        unpacked_user_id.append(uid)
    
    # Pad the sequence with less length in a batch
    inputs_padded = torch.nn.utils.rnn.pad_sequence(inputs_batch, padding_value=0, batch_first=True) 
    labels_padded = torch.nn.utils.rnn.pad_sequence(labels_batch, padding_value=0, batch_first=True)
    positions_padded = torch.nn.utils.rnn.pad_sequence(positions_batch, padding_value=0, batch_first=True) 
    label_positions_padded = torch.nn.utils.rnn.pad_sequence(label_positions_batch, padding_value=0, batch_first=True)
    
    return torch.tensor(unpacked_user_id), inputs_padded, labels_padded, positions_padded, label_positions_padded

BATCH_SIZE_train = (len(train_dataset)//len(grouped_data_train))*10
BATCH_SIZE_test = 1 # BATCH_SIZE_train

train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE_train, shuffle=True,  collate_fn=collate_fn)
test_dataloader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE_test, shuffle=False, collate_fn=collate_fn)

print(f"{len(train_dataset)} Training data and {len(test_dataset)} Testing data loaded ... with train batch size being {BATCH_SIZE_train} and with test batch size being {BATCH_SIZE_test}!")

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

print("PyTorch Built-in LSTM loaded!")

def decode_token_to_cell(token_id, num_cells_per_row):
    cell_x = token_id % num_cells_per_row + 1
    cell_y = token_id // num_cells_per_row + 1 
    return cell_x, cell_y

def decode_trajectory(traj, num_cells_per_row=200):
    decoded_traj = [decode_token_to_cell(token_id, num_cells_per_row) for token_id in traj]
    return decoded_traj

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

        # Iterate over the batch
        for i in range(labels.size(0)):
            # Retrieve true trajectory and predicted trajectory
            true_traj = labels[i].cpu().numpy()
            pred_traj = predicted[i].cpu().numpy()

            # Decode true trajectory and predicted trajectory
            decoded_true_traj = np.array(decode_trajectory(true_traj))
            decoded_pred_traj = np.array(decode_trajectory(pred_traj))

            # Euclidean Distance Calcualtion
            euclidean_distances = np.linalg.norm(decoded_true_traj - decoded_pred_traj, axis=1)
            total_distance += np.sum(euclidean_distances)
            total_trajectories += 1

            # Apply threshold
            if np.sum(euclidean_distances) <= threshold:
                correct_trajectories += 1
    
    # Calculate loss
    avg_loss = total_loss / len(dataloader)

    # Calculate accuracy
    avg_euclidean_distance = total_distance / total_trajectories
    accuracy = correct_trajectories / total_trajectories

    print(f"Average Euclidean Distance Difference: {avg_euclidean_distance:.4f}, Accuracy: {accuracy:.4f}") 
    
    return avg_loss, avg_euclidean_distance, accuracy

def inference(model, dataloader, device, threshold=(1+math.sqrt(2))):
    model.eval() 
    total_distance = 0.0
    total_trajectories = 0
    correct_trajectories = 0
    
    with torch.no_grad():  
        for inputs, labels, _, _ in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)  
            _, predicted = outputs.max(2)  
            
            # Iterate over the batch
            for i in range(labels.size(0)):
                true_traj = labels[i].cpu().numpy()
                pred_traj = predicted[i].cpu().numpy()

                # Decode true trajectory and predicted trajectory
                decoded_true_traj = np.array(decode_trajectory(true_traj))
                decoded_pred_traj = np.array(decode_trajectory(pred_traj))

                # Euclidean Distance Calcualtion
                euclidean_distances = np.linalg.norm(decoded_true_traj - decoded_pred_traj, axis=1)
                total_distance += np.sum(euclidean_distances)
                total_trajectories += 1

                # Apply threshold
                if euclidean_distances <= threshold:
                    correct_trajectories += 1

    # Calculate accuracy
    avg_euclidean_distance = total_distance / total_trajectories 
    accuracy = correct_trajectories / total_trajectories

    print(f"Average Euclidean Distance Difference: {avg_euclidean_distance:.4f}, Accuracy: {accuracy:.4f}")

    return accuracy

def recursive_inference_per_user(model, dataloader, device, input_size, output_size, total_outputs=192):
    model.eval()
    all_user_predictions = {} 

    with torch.no_grad():
        start = False

        for user_id, inputs, _, _, _ in dataloader:
            
            # Extract info from dataloader
            user_id = user_id.item()
            inputs = inputs.to(device)

            predictions = []
            current_input = []

            if(not start):
                current_input = inputs
                start = True

            # Generate predictions recursively for the current user
            while len(predictions) < total_outputs:

                outputs = model(current_input)
                
                # Get the index of the max log-probability
                _, predicted = outputs.max(2)
                
                # Store prediction
                predictions.extend(predicted.cpu().numpy().tolist()[0]) 

                # Prepare for the next prediction
                current_input = predicted

            all_user_predictions[user_id] = predictions[:total_outputs]
            start = False
    
    return all_user_predictions    

def measure_accuracy_recursive_inference(all_user_predictions, test_data, real_test_data, threshold=(1+math.sqrt(2))):
    total_distance = 0.0
    total_trajectories = 0
    correct_trajectories = 0

    for test_uid, trajectory in all_user_predictions.items():
        # Load prediction and label
        temp_test_data = test_data[(test_data['uid']==test_uid) & (test_data['x']==999)]
        test_data_day = temp_test_data['d']
        test_data_time = temp_test_data['t']

        temp_real_test_data = real_test_data[(real_test_data['uid']==test_uid) & (real_test_data['d'].isin(test_data_day)) & (real_test_data['t'].isin(test_data_time))]
        predicted_test_data = trajectory[:len(temp_real_test_data)]
        
        # # Decode true trajectory and predicted trajectory
        decoded_true_traj = temp_real_test_data[['x', 'y']].to_numpy()
        decoded_pred_traj = np.array(decode_trajectory(predicted_test_data))

        print("True Trajectory:")
        print(decoded_true_traj)
        print("Predicted Trajectory:")
        print(decoded_pred_traj)

        # Euclidean Distance Calculate
        euclidean_distances = np.linalg.norm(decoded_true_traj - decoded_pred_traj, axis=1)
        print("Euclidean Distance Difference:", euclidean_distances)
        print()

        total_distance += np.sum(euclidean_distances)

        # Apply threshold
        for euclidean_distance in euclidean_distances:
            if euclidean_distance <= threshold:
                correct_trajectories += 1
            total_trajectories += 1

    # Calculate accuracy
    avg_euclidean_distance = total_distance / total_trajectories 
    accuracy = correct_trajectories / total_trajectories

    print(f"Average Euclidean Distance Difference: {avg_euclidean_distance:.4f}, Accuracy: {accuracy:.4f}")
    return accuracy


def train_model(model, dataloader, device, epochs, learning_rate):
    for epoch in range(epochs):
        print(f"Test: Epoch {epoch+1}")
        train(model, dataloader, device, learning_rate)
        ## print("Inference")
        ## inference(model, test_dataloader, device)

print("Start training process!")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCH_NUM = 3
lstm = LSTMModel(loc_size=40000, 
                 output_size=output_size, 
                 embed_dim=64, 
                 hidden_size=64, 
                 num_layers=1, 
                 device=device)
lstm.to(device)
train_model(lstm, train_dataloader, device, EPOCH_NUM, 0.001)

# Autoregressive Inference
all_user_predictions = recursive_inference_per_user(model, test_dataloader, device, input_size, output_size, total_outputs=192)
print("Predicted data loaded!")

# Output accuracy
test_data = pd.read_csv('test.csv') # file with 999 (unknown number)
real_test_data = pd.read_csv('yjmob100k-dataset1.csv.gz', compression='gzip') # file with actual location info
print("Actual data loaded. Ready to measure accuracy!")

measure_accuracy_recursive_inference(all_user_predictions, test_data, real_test_data)