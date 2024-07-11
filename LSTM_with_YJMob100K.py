import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
import math

# Load data with users from yjmob1
df_train = pd.read_csv('train.csv')
## df_test = pd.read_csv('test.csv')

# Group data by uid
grouped_data_train = [group for _, group in df_train.groupby('uid')]
## grouped_data_test  = [group for _, group in df_test.groupby('uid')]

# Adjust input and output size here
input_size  = 48 * 2 # now: 2 days # previous:  math.floor(584 * 0.8) = 467
output_size = 48     # now: 1 day  # previous:  math.ceil(584 * 0.2) = 117

class TrajectoryDataset(Dataset):
    def __init__(self, grouped_data, input_size, output_size):
        self.data = []
        for group in grouped_data:
            xy = group['combined_xy'].values.tolist()
            t = group['t'].values.tolist()

            ## One-time apporach
            # self.data.append((xy[0:input_size], xy[input_size:(input_size+output_size)], t[0:input_size], t[input_size:(input_size+output_size)]))
            
            # Sliding window approach
            window_size = input_size + output_size
            for i in range(0, len(group)-window_size+1, window_size):
                input_end = i + input_size
                self.data.append((xy[i:input_end], xy[input_end:(input_end+output_size)], t[i:input_end], t[input_end:(input_end+output_size)]))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        inputs, labels, positions, label_positions = self.data[idx]
        return torch.tensor(inputs), torch.tensor(labels), torch.tensor(positions), torch.tensor(label_positions)

train_dataset = TrajectoryDataset(grouped_data_train, input_size, output_size)
## test_dataset = TrajectoryDataset(grouped_data_test, input_size, output_size)

# Clutch train and test datasets into dataloaders
def collate_fn(batch):
    # Unzip all batch
    inputs_batch, labels_batch, positions_batch, label_positions_batch = zip(*batch)
    
    # Pad the sequence with less length in a batch
    inputs_padded = torch.nn.utils.rnn.pad_sequence(inputs_batch, padding_value=0, batch_first=True) 
    labels_padded = torch.nn.utils.rnn.pad_sequence(labels_batch, padding_value=0, batch_first=True)
    positions_padded = torch.nn.utils.rnn.pad_sequence(positions_batch, padding_value=0, batch_first=True) 
    label_positions_padded = torch.nn.utils.rnn.pad_sequence(label_positions_batch, padding_value=0, batch_first=True)
    
    return inputs_padded, labels_padded, positions_padded, label_positions_padded

BATCH_SIZE = (len(train_dataset)//len(grouped_data_train))*10

train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
## test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

print(f"{len(train_dataset)} training data loaded with batch_size being {BATCH_SIZE}!") # and {len(test_dataset)} Testing data 

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

print("PyTorch's Built-in LSTM loaded!")

def decode_token_to_cell(token_id, num_cells_per_row):
    cell_x = (token_id % num_cells_per_row) + 1
    cell_y = (token_id // num_cells_per_row) + 1
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
    
    for inputs, labels, _, _ in dataloader: 
        inputs, labels = inputs.to(device), labels.to(device)
        
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

def train_model(model, dataloader, device, epochs, learning_rate):
    for epoch in range(epochs):
        print(f"Test: Epoch {epoch+1}")
        train(model, dataloader, device, learning_rate)
        ## print("Inference")
        ## inference(model, test_dataloader, device)
        print()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
lstm = LSTMModel(loc_size=40000, output_size=output_size, embed_dim=64, hidden_size=64, num_layers=1, device=device)
lstm.to(device)

print("Start training process!")
EPOCH_NUM = 3
train_model(lstm, train_dataloader, device, EPOCH_NUM, 0.001)