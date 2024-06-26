import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader

# Load data with 10k users from yjmob1

df_train = pd.read_csv('train.csv')
df_test = pd.read_csv('test.csv')

# Group data by uid

grouped_data_train = [group for _, group in df_train.groupby('uid')]
grouped_data_test  = [group for _, group in df_test.groupby('uid')]

# For the training sesion, the former 48 data would be used to predict the latter 48 data 

class TrajectoryDataset(Dataset):
    def __init__(self, grouped_data, step_size):
        self.data = []
        for group in grouped_data:
            xy = group['combined_xy'].values.tolist()
            t = group['t'].values.tolist()
            window_size = step_size * 2
            for i in range(0, len(group) - window_size + 1, step_size):
                input_end = i + step_size
                predict_end = input_end + step_size
                self.data.append((xy[i:input_end], xy[input_end:predict_end], t[i:input_end], t[input_end:predict_end]))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        inputs, labels, positions, label_positions = self.data[idx]
        return torch.tensor(inputs), torch.tensor(labels), torch.tensor(positions), torch.tensor(label_positions)

train_dataset = TrajectoryDataset(grouped_data_train, 48)
test_dataset = TrajectoryDataset(grouped_data_test, 48)

# clutch train and test datasets into dataloaders

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
test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

print(f"{len(train_dataset)} Training and {len(test_dataset)} Testing data loaded with batch_size being {BATCH_SIZE}!")

class LSTMModel(nn.Module):
    def __init__(self, loc_size, embed_dim, hidden_size, num_layers, device):
        super(LSTMModel, self).__init__()
        self.input_embedding = nn.Embedding(loc_size, embed_dim)
        self.lstm = nn.LSTM(input_size=embed_dim, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, loc_size)
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.device = device

    def forward(self, x):
        x = self.input_embedding(x) # positioanl embedding
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device) # initialize hidden state
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device) # initialize cell state

        # LSTM
        out, _ = self.lstm(x, (h0, c0)) 
        out = self.fc(out)
        return out

print("PyTorch's Built-in LSTM loaded!")

def train(model, dataloader, device, learning_rate):
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=learning_rate)
    
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    
    for inputs, labels, _, _ in dataloader:  # Assume dataloader yields only inputs and labels
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs.view(-1, outputs.size(-1)), labels.view(-1))
        
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # Calculate accuracy
        _, predicted = outputs.max(2) # Get the index of the max log-probability
        total_correct += (predicted == labels).sum().item()
        total_samples += labels.numel()
    
    avg_loss = total_loss / len(dataloader)
    accuracy = total_correct / total_samples
    
    return avg_loss, accuracy

def inference(model, dataloader, device):
    model.eval()  
    total_correct = 0
    total_samples = 0
    
    with torch.no_grad():  
        for inputs, labels, _, _ in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)  
            _, predicted = outputs.max(2)  
            
            total_correct += (predicted == labels).sum().item()  
            total_samples += labels.numel()  

    accuracy = total_correct / total_samples  
    print(f"Total Correct: {total_correct}, Total Samples: {total_samples}, Accuracy: {accuracy:.2f}")

    return accuracy

def train_model(model, dataloader, device, epochs, learning_rate):
    for epoch in range(epochs):
        avg_loss, accuracy = train(model, dataloader, device, learning_rate)
        print(f"Epoch {epoch}, Average Loss: {avg_loss}, Accuracy: {accuracy:.4f}")

        inference(model, test_dataloader, device) # this is here to see if model overfits

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
lstm = LSTMModel(loc_size=40000, embed_dim=256, hidden_size=256, num_layers=2, device=device)
lstm.to(device)

print("Start training process!")
EPOCH_NUM = 3
train_model(lstm, train_dataloader, device, EPOCH_NUM, 0.001)

print ("Start inference process!")
lstm_accuracy = inference(lstm, test_dataloader, device)