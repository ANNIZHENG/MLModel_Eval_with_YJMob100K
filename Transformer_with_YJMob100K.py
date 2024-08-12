import pandas as pd
import numpy as np
import math
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader, Sampler
from torch.nn.utils.rnn import pad_sequence

# Load data with users from yjmob1
# df_train = pd.read_csv('train.csv')
df_test  = pd.read_csv('test.csv')
df_train = df_test
df_true_test = pd.read_csv('true_test.csv')

class TrajectoryDataset(Dataset):
    def __init__(self, all_data, input_size, output_size):
        self.data = []
        window_size = input_size + output_size
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

# Adjust input and output size here
input_size  = 192
output_size = 48

train_dataset = TrajectoryDataset(df_train, input_size, output_size)
test_dataset = TrajectoryDataset(df_test, input_size, output_size)

class UserGroupSampler(Sampler):
    def __init__(self, dataset):
        self.indices_by_user = {}
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

def collate_fn(batch):
    user_ids, inputs_batch, labels_batch, positions_batch, label_positions_batch = zip(*batch)
    inputs_padded = pad_sequence(inputs_batch, batch_first=True, padding_value=0)
    labels_padded = pad_sequence(labels_batch, batch_first=True, padding_value=0)
    positions_padded = pad_sequence(positions_batch, batch_first=True, padding_value=0)
    label_positions_padded = pad_sequence(label_positions_batch, batch_first=True, padding_value=0)
    ## return user_ids, inputs_padded, labels_padded, positions_padded, label_positions_padded
    return user_ids[0].clone().detach(), inputs_padded, labels_padded, positions_padded, label_positions_padded

train_sampler = UserGroupSampler(train_dataset) # group data of the same user id on the same batch
train_dataloader = DataLoader(train_dataset, batch_sampler=train_sampler, collate_fn=collate_fn)

test_sampler = UserGroupSampler(test_dataset)
test_dataloader = DataLoader(test_dataset, batch_sampler=test_sampler, collate_fn=collate_fn)

print("Training data and Testing data loaded!")

# Assuming the same DataLoader setup and Dataset class as before

class TransformerModel(nn.Module):
    def __init__(self, loc_size, time_size_input, time_size_output, embed_dim, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, dropout, device):
        super(TransformerModel, self).__init__()
        self.device = device
        self.input_embedding = nn.Embedding(loc_size, embed_dim).to(device)
        self.pos_encoder = nn.Embedding(time_size_input, embed_dim).to(device)
        self.pos_decoder = nn.Embedding(time_size_output, embed_dim).to(device)
        self.transformer = nn.Transformer(d_model=embed_dim, nhead=nhead, num_encoder_layers=num_encoder_layers, num_decoder_layers=num_decoder_layers, dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True).to(device)
        self.fc_out = nn.Linear(embed_dim, loc_size).to(device)

    def forward(self, src_seq, src_pos, trg_seq, trg_pos):
        src_emb = self.input_embedding(src_seq) + self.pos_encoder(src_pos)
        trg_emb = self.input_embedding(trg_seq) + self.pos_decoder(trg_pos)
        output = self.transformer(src_emb, trg_emb)
        return self.fc_out(output)

# Instantiate and use the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TransformerModel(
    loc_size=40000, 
    time_size_input=192, 
    time_size_output=48, 
    embed_dim=512, 
    nhead=8, 
    num_encoder_layers=6, 
    num_decoder_layers=6, 
    dim_feedforward=2048, 
    dropout=0.1, 
    device=device
)
model.to(device)

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
    
    for _, inputs, labels, positions, label_positions in dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        positions = positions.to(device)
        label_positions = label_positions.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs, positions, labels, label_positions)
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

def train_model(model, dataloader, device, epochs, learning_rate):
    for epoch in range(epochs):
        print(f"Train Epoch {epoch+1}")
        train(model, dataloader, device, learning_rate)

print("Start Training Process!")

train_model(model=model, dataloader=test_dataloader, device=device, epochs=5, learning_rate=0.001)