import gzip
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import optuna
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import random

yjmob1 = 'yjmob100k-dataset1.csv.gz'
yjmob_df = pd.read_csv(yjmob1, compression='gzip')

# Retrieve all ids
uids = yjmob_df['uid'].unique()

# Just to reduce memory space
rand_indicies = [random.randint(0, len(uids)) for _ in range(10000)]
selected_uids = [uid for uid in uids[rand_indicies]]

df = yjmob_df[yjmob_df['uid'].isin(selected_uids)] 

# Time
# df['combined_t'] = df['d']*47+df['t']

# Location
def spatial_token(x, y):
    return (x-1)+(y-1)*200

df['combined_xy'] = df.apply(lambda row: spatial_token(row['x'], row['y']), axis=1)

# 8:2 split
train_uids, test_uids = train_test_split(selected_uids, test_size=0.2, random_state=42)

BATCH_SIZE = 1 # 25
STEP_SIZE = 100

def generate_sequences(data, data_t):
    return torch.tensor(data[:STEP_SIZE]),torch.tensor(data[STEP_SIZE]),\
                torch.tensor(data_t[:STEP_SIZE]),torch.tensor(data_t[STEP_SIZE])

# Group data by uid
grouped_data_train = [group for _, group in df_train.groupby('uid')]
grouped_data_test = [group for _, group in df_test.groupby('uid')]

class TrajectoryDataset(Dataset):
    def __init__(self, grouped_data):
        self.data = grouped_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data_for_uid = self.data[idx]
        inputs, labels, positions, label_positions = generate_sequences(
                                                         data_for_uid['combined_xy'].values.tolist(),
                                                         data_for_uid['t'].values.tolist())
        return inputs, labels, positions, label_positions

train_dataset = TrajectoryDataset(grouped_data_train)
test_dataset = TrajectoryDataset(grouped_data_test)

def collate_fn(batch):
    # Unzip all batch
    inputs_batch, labels_batch, positions_batch, label_positions_batch = zip(*batch)
    
    # Pad the sequence with less length in a batch
    inputs_padded = torch.nn.utils.rnn.pad_sequence(inputs_batch, padding_value=0.0, batch_first=True)
    labels_padded = torch.tensor(np.array(labels_batch))
    positions_padded = torch.nn.utils.rnn.pad_sequence(positions_batch, padding_value=0.0, batch_first=True)
    label_positions_padded = torch.tensor(np.array(label_positions_batch))
    
    # Doing Addition here
    # return inputs_padded+positions_padded, labels_padded+label_positions_padded
    return inputs_padded, labels_padded

train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

class LSTMModel(nn.Module):
    def __init__(self, input_dim, embed_dim, layer_dim, output_dim):
        super(LSTMModel, self).__init__()
        self.embed_dim = embed_dim
        self.layer_dim = layer_dim
        self.lstm = nn.LSTM(input_dim, embed_dim, layer_dim, batch_first=True)
        self.fc = nn.Linear(embed_dim, output_dim)
    
    def forward(self, x):
        # Initialize hidden and cell states
        h0 = torch.zeros(self.layer_dim, x.size(0), self.embed_dim).to(x.device)
        c0 = torch.zeros(self.layer_dim, x.size(0), self.embed_dim).to(x.device)
        
        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))
        
        # Taking the output of the last sequence step
        out = self.fc(out[:, -1, :])
        return out
    
# Model related param
EMBED_DIM = 256
INPUT_DIM = 1
LAYER_DIM = 1
NUM_CLASS = 40000 # +48 # 200*200 grid loc + 48 time

model = LSTMModel(input_dim=INPUT_DIM, 
                  embed_dim=EMBED_DIM, 
                  layer_dim=LAYER_DIM, 
                  output_dim=NUM_CLASS)
optimizer = optim.Adam(model.parameters(), lr=0.0013)
criterion = nn.CrossEntropyLoss()

epochs = 10

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

batch_sizes = [int(i) for i in range(5,101,5)]
batch_sizes.append(1)
batch_sizes = sorted(batch_sizes)

def objective(trial):
    # Set up dataloader
    batch_size = trial.suggest_categorical('batch_size', batch_sizes)  ##
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    
    # Model Parameters
    NUM_CLASS = 40000
    # STEP_SIZE = 100
    # EMBED_DIM = trial.suggest_categorical('embed_dim', [64, 128, 256, 512])
    EMBED_DIM = 256
    LAYER_DIM = trial.suggest_int('layer_dim', 1, 6)  ##
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Train the model
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-1)
    
    # Model instantiation
    model = LSTMModel(input_dim=1,
                      embed_dim=EMBED_DIM, 
                      layer_dim=LAYER_DIM,
                      output_dim=NUM_CLASS)
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    
    # Training loop
    total_loss = 0
    total_samples = 0
    for _ in range(5):
        model.train()
        for inputs, labels in train_dataloader:
            inputs = inputs.float().unsqueeze(-1)
            labels = labels.long()
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * labels.size(0)
            total_samples += labels.size(0)
    
    final_avg_loss = total_loss / total_samples
    return final_avg_loss

# Hyperparameter tuning

# Create a study object and optimize the objective function
study = optuna.create_study()
study.optimize(objective, n_trials=50)

# Result
print('Best parameters:', study.best_params)
print('Best loss:', study.best_value)