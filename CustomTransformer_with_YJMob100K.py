import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
import random

# Load data with 10k users from yjmob1

df_train = pd.read_csv('train.csv')
df_test = pd.read_csv('test.csv')

# Group data by uid

grouped_data_train = [group for _, group in df_train.groupby('uid')]
grouped_data_test  = [group for _, group in df_test.groupby('uid')]

# For the training sesion, the former TIME_POSITION_SIZE data would be used to predict the latter TIME_POSITION_SIZE data 

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
    
# We are going to take `TIME_POSITION_SIZE` time as input to predict `TIME_POSITION_SIZE` time
TIME_POSITION_SIZE = 96
train_dataset = TrajectoryDataset(grouped_data_train, TIME_POSITION_SIZE)
test_dataset = TrajectoryDataset(grouped_data_test, TIME_POSITION_SIZE)

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

# Time = Positional Encoding = Time Embedding + Sequential Encoding

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
            raise ValueError(f"Input sequence length ({x.size(1)}) \
            is greater than the number of positional encodings available ({self.pe.size(0)})")
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
        attn_output, attn_output_weights = self.multihead_attn(query, key, value, attn_mask=attn_mask)
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
    def __init__(self, loc_size, time_size, embed_dim, num_layers, num_heads, 
                 device, forward_expansion, dropout_rate):
        super(Transformer, self).__init__()
        
        self.encoder = Encoder(loc_size, time_size, embed_dim, num_layers, num_heads, device, forward_expansion, dropout_rate)
        self.decoder = Decoder(loc_size, time_size, embed_dim, num_layers, num_heads, device, forward_expansion, dropout_rate)
        self.device = device
    
    def forward(self, src_seq, src_pos, trg_seq, trg_pos):
        # Encode source
        enc_out = self.encoder(src_seq, src_pos)
        # Decode target
        dec_out = self.decoder(trg_seq, trg_pos, enc_out)
        return dec_out
    
print("Custom Transformer loaded!")

def train(model, dataloader, device, learning_rate):
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=learning_rate)
    
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    
    for inputs, labels, positions, label_positions in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        positions, label_positions = positions.to(device), label_positions.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs, positions, labels, label_positions)
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
        for inputs, labels, positions, label_positions in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            positions, label_positions = positions.to(device), label_positions.to(device)

            outputs = model(inputs, positions, positions, label_positions) 
            _, predicted = outputs.max(2)
            
            total_correct += (predicted == labels).sum().item()
            total_samples += labels.numel()

    accuracy = total_correct / total_samples
    print(f"Inference: Total Correct: {total_correct}, Total Samples: {total_samples}, Accuracy: {accuracy:.4f}")

    return accuracy

def train_model(model, dataloader, device, epochs, learning_rate):
    for epoch in range(epochs):
        avg_loss, accuracy = train(model, dataloader, device, learning_rate)
        print(f"Test: Epoch {epoch}, Average Loss: {avg_loss}, Accuracy: {accuracy:.4f}")
        inference(model, test_dataloader, device)
        print()

print("Start training process!")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCH_NUM = 5
transformer = Transformer(loc_size=40000, 
                          time_size=TIME_POSITION_SIZE,
                          embed_dim=64,
                          num_layers=1,
                          num_heads=4,
                          device=device,
                          forward_expansion=4,
                          dropout_rate=0.3)
transformer.to(device)
train_model(transformer, train_dataloader, device, epochs=EPOCH_NUM, learning_rate=0.001)

# print ("Start inference process!")
# transformer_accuracy = inference(transformer, test_dataloader, device)