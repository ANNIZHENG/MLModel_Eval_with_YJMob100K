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

# Time data being the positional encoded data
class PositionalEncoding(nn.Module):
    def __init__(self, max_len, embedding_dim):
        super(PositionalEncoding, self).__init__()
        self.embedding_dim = embedding_dim
        self.dropout = nn.Dropout(p=0.1)
        
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embedding_dim, 2) * (-np.log(10000.0) / embedding_dim))
        pe = torch.zeros(max_len, 1, embedding_dim)
        pe[:, 0, 0::2] = torch.sin(position.float() * div_term)
        pe[:, 0, 1::2] = torch.cos(position.float() * div_term)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        if x.size(1) > self.pe.size(0):
            raise ValueError(f"Input sequence length ({x.size(1)}) is greater than the number of positional encodings available ({self.pe.size(0)})")
        x = x + self.pe[:x.size(1)].squeeze(1).expand_as(x)
        return self.dropout(x)
    
NUM_HEADS = 8

class MultiHeadAttentionModule(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout_rate):
        super(MultiHeadAttentionModule, self).__init__()
        self.multihead_attn = nn.MultiheadAttention(embed_dim, num_heads)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, query, key, value, attn_mask=None, key_padding_mask=None, need_weights=True):
        # Transpose from [batch size, seq length, embed dim] to [seq length, batch size, embed dim]
        query = query.transpose(0, 1)
        key = key.transpose(0, 1)
        value = value.transpose(0, 1)
        
        # Apply multihead attention
        attn_output, _ = self.multihead_attn(query, key, value, attn_mask=attn_mask)
        attn_output = self.dropout(attn_output)
        return attn_output.transpose(0, 1)
    
class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, forward_expansion, dropout_rate):
        super(TransformerBlock, self).__init__()
        
        # Attention Layer
        self.attention = MultiHeadAttentionModule(embed_dim, num_heads, dropout_rate)
        
        # Normalization 1 
        self.norm1 = nn.LayerNorm(embed_dim)
        
        # Feed-Forward
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, forward_expansion * embed_dim), 
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_dim, embed_dim)
        )
        
        # Normalization 2
        self.norm2 = nn.LayerNorm(embed_dim)
        
        # Dropout
        self.dropout = nn.Dropout(dropout_rate)
        
    def forward(self, query, key, value):
        attn_output = self.attention(query, key, value)
        x = self.norm1(attn_output + query)
        forward = self.feed_forward(x)
        out = self.norm2(self.dropout(forward + x))
        return out
    
class Encoder(nn.Module):
    def __init__(self, loc_size, time_size, embed_dim, num_layers, num_heads, device, forward_expansion, dropout_rate):
        super(Encoder, self).__init__()
        self.embed_dim = embed_dim
        self.device = device
        
        self.input_embedding = nn.Embedding(loc_size, embed_dim).to(device)
        self.position_embedding = nn.Embedding(time_size, embed_dim).to(device)
        self.positional_encoding = PositionalEncoding(time_size, embed_dim).to(device)
        
        self.layers = nn.ModuleList([
            TransformerBlock(
                embed_dim,
                num_heads,
                forward_expansion=forward_expansion,
                dropout_rate=dropout_rate
            ) for _ in range(num_layers)
        ])
        
    def forward(self, inputs, positions):
        # Input Embedding
        space = self.input_embedding(inputs)

        # Positional Encoding
        positions = self.position_embedding(positions) 
        time = self.positional_encoding(positions)
        
        # Addition
        out = space + time

        # Remind: Transformer Block = Multi-Head Attention + Norm + Feed Forward + Norm
        for layer in self.layers:
            out = layer(out, out, out)
        return out
    
class DecoderBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, forward_expansion, device, dropout_rate): 
        super(DecoderBlock, self).__init__()
        self.attention = MultiHeadAttentionModule(embed_dim, num_heads, dropout_rate)
        self.norm = nn.LayerNorm(embed_dim)
        self.transformer_block = TransformerBlock(embed_dim, num_heads, forward_expansion, dropout_rate)
        
    def forward(self, x, key, value, attn_mask=None):
        attention = self.attention(x,key,value,attn_mask)
        query = self.norm(attention + x)
        out = self.transformer_block(query, key, value)
        return out

class Decoder(nn.Module):
    def __init__(self, loc_size, time_size, embed_dim, num_layers, num_heads, device, forward_expansion, dropout_rate):
        super(Decoder, self).__init__()
        self.embed_dim = embed_dim
        self.device = device
        self.input_embedding = nn.Embedding(loc_size, embed_dim).to(device)
        self.position_embedding = nn.Embedding(time_size, embed_dim).to(device)
        self.positional_encoding = PositionalEncoding(time_size, embed_dim).to(device)
        self.layers = nn.ModuleList([
            DecoderBlock(
                embed_dim,
                num_heads,
                forward_expansion=forward_expansion,
                device=device,
                dropout_rate=dropout_rate
            ) for _ in range(num_layers)
        ])
        self.fc_out = nn.Linear(embed_dim, loc_size)
        
    def forward(self, output, output_position, enc_out):
        # Input Embedding
        space = self.input_embedding(output)

        # Positional Encoding
        positions = self.position_embedding(output_position)
        time = self.positional_encoding(positions)

        # Addition
        out = space + time

        # Decoding process
        for layer in self.layers:
            out = layer(out, enc_out, enc_out, attn_mask=None)

        # Output through final linear layer
        out = self.fc_out(out)
        return out
    
class Transformer(nn.Module):
    def __init__(self, loc_size, time_size_input, time_size_output, embed_dim, num_layers, num_heads, device, forward_expansion, dropout_rate):
        super(Transformer, self).__init__()
        self.encoder = Encoder(loc_size, time_size_input, embed_dim, num_layers, num_heads, device, forward_expansion, dropout_rate)
        self.decoder = Decoder(loc_size, time_size_output, embed_dim, num_layers, num_heads, device, forward_expansion, dropout_rate)
        self.device = device

    def forward(self, src_seq, src_pos, trg_seq, trg_pos, decode):
        if (decode):
            enc_out = self.encoder(src_seq, src_pos)
            dec_out = self.decoder(trg_seq, trg_pos, enc_out)
            return dec_out
        else:
            enc_out = self.encoder(src_seq, src_pos)
            return enc_out
    
print("Custom Transformer loaded!")

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
        outputs = model(inputs, positions, labels, label_positions, True)
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

def train_model(model, dataloader, device, epochs, learning_rate):
    for epoch in range(epochs):
        print(f"Train Epoch {epoch+1}")
        train(model, dataloader, device, learning_rate)

print("Start training process!")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Transformer(loc_size=40000, time_size_input=input_size, time_size_output=output_size, embed_dim=64, num_layers=1, num_heads=4, device=device, forward_expansion=4, dropout_rate=0.0)
model.to(device)
train_model(model, train_dataloader, device, epochs=5, learning_rate=0.001)

# Exapnd prediction to prepare to correspond to ground truth
def expand_predictions(predicted_locs, predicted_times, max_time=47):
    expanded_locs = []
    expanded_times = list(range(max_time + 1))
    current_loc = predicted_locs[0]
    loc_dict = dict(zip(predicted_times, predicted_locs))
    for time in expanded_times:
        if time in loc_dict:
            current_loc = loc_dict[time]
        expanded_locs.append(current_loc)
    return expanded_locs, expanded_times

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

    # Accuracy measure based on Euclidean Distance difference
    euclidean_distances = np.linalg.norm(true_locs - matched_locs, axis=1)
    total_distance += np.sum(euclidean_distances)

    for euclidean_distance in euclidean_distances:
        total_location += 1
        if (euclidean_distance < threshold):
            correct_location += 1
    
    # avg_euclidean_distance = total_distance / total_location 
    # accuracy = correct_location / total_location
    # print(f"User ID: {user_id}, Average Euclidean Distance Difference: {avg_euclidean_distance:.4f}, Accuracy: {accuracy:.4f}")

    return matched_locs, total_distance, total_location, correct_location

def recursive_inference_per_user(model, dataloader, device, true_data):
    # Measurement used for total accuracy calculation
    total_distances = 0.0 # total distance off
    total_locations = 0 # total num of location to be predicted
    correct_locations = 0 # total num of locations that are correctly predicted

    model.eval()
    
    predictions = {} # Store the 15-day predicted values per user
    predictions_time = {} # Store the corresponding 15-day time values per user
    
    with torch.no_grad():
        for user_id, inputs, labels, positions, label_positions in dataloader:    

            user_id = user_id.item()
            inputs = inputs.to(device)
            labels = labels.to(device)
            positions = positions.to(device)
            label_positions = label_positions.to(device)

            # Load Ground Truth Data
            true_data_by_uid = true_data[true_data['uid']==user_id]
            true_locs = np.array(list(zip(true_data_by_uid['x'], true_data_by_uid['y']))) # ground truth location
            true_times = true_data_by_uid['t'].to_list() # ground truth time
            num_predictions = len(true_data_by_uid) # number of needed prediction
            
            # Store predictions and times for the current user
            user_predictions = []
            user_predictions_time = []

            # Initial Prediction
            outputs = model(inputs, positions, labels, label_positions, True)
            _, predicted = outputs.max(2)  # Get the index of the max log-probability
            
            # Set up for Autoregressive test
            current_input = inputs
            current_positions = positions
            current_label = predicted
            current_label_positions = label_positions
            
            for i in range(current_input.size(0)):
                user_predictions.extend(predicted[i].cpu().numpy())
                user_predictions_time.extend(label_positions[i].cpu().numpy()) # user_predictions_time.extend(positions[i].cpu().numpy())

            # Autoregressive Prediction
            while (len(user_predictions) < num_predictions):
                # Initialize batch-related variables for prediction
                new_inputs = []
                new_positions = []
                new_labels = []
                new_label_positions = []
                
                for i in range(current_input.size(0)):
                    true_traj = current_input[i].cpu().numpy() # len: 192
                    pred_traj = current_label[i].cpu().numpy() # len: 48

                    # Concatenate and truncate to create new input
                    new_input_traj = torch.tensor(np.concatenate((true_traj[48:], pred_traj))).to(device) # Concatenate prediction with the previous input
                    new_position = current_positions[i].to(device) # the KNOWN time data for the initial prediction
                    new_label_position = current_label_positions[i].to(device) # the KNOWN label's time data for the initial prediction

                    new_inputs.append(new_input_traj)
                    new_positions.append(new_position)
                    new_labels.append(current_label[i])
                    new_label_positions.append(new_label_position)
                
                # Stack to form a new batch
                current_input = torch.stack(new_inputs)
                current_positions = torch.stack(new_positions)
                current_label = torch.stack(new_labels)
                current_label_positions = torch.stack(new_label_positions)
                
                # New Prediction
                new_outputs = model(current_input, current_positions, current_label, current_label_positions, True) 
                _, predicted = new_outputs.max(2)
                
                for i in range(current_input.size(0)):
                    user_predictions.extend(predicted[i].cpu().numpy())
                    user_predictions_time.extend(current_label_positions[i].cpu().numpy())

                current_label = predicted
            
            # Measure accuracy for each user's prediction
            matched_locs, total_distance, total_location, correct_location = accuracy_measure(user_id, user_predictions[:num_predictions], user_predictions_time[:num_predictions], true_locs, true_times)

            # Store predictions and times for the user
            predictions[user_id] = matched_locs
            predictions_time[user_id] = true_times

            # Record the total distance
            total_distances += total_distance
            total_locations += total_location
            correct_locations += correct_location

    avg_euclidean_distance = total_distances / total_locations
    accuracy = correct_locations / total_locations

    print(f"Total Users' Adjusted Euclidean Distance Difference: {avg_euclidean_distance:.4f}, Accuracy: {accuracy:.4f}")

    return avg_euclidean_distance, accuracy, predictions, predictions_time

# Autoregressive Inference
print("Test")
_, _, predictions, predictions_time = recursive_inference_per_user(model, test_dataloader, device, df_true_test)

print(predictions)
print()
print()
print(predictions_time)