import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import math
import numpy as np
import pandas as pd


# Transformer model to work with distance matrix
class TransformerModel(nn.Module):
    def __init__(self, input_dim, nhead, nhid, nlayers, d_model=64, dropout=0.1):
        super(TransformerModel, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # positional encoding (embedding layer)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        
        # Transformer Encoder Layer
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, nhid, dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, nlayers)
        
        # Final linear layer 
        # i don't know if we need this
        self.fc_out = nn.Linear(d_model, input_dim)

    def forward(self, src):
        # Positional encoding
        src = self.pos_encoder(src)

        # transformer layers
        output = self.transformer_encoder(src)
        
        # Final output layer
        output = self.fc_out(output)
        return output

# Positional Encoding for Transformer
class PositionalEncoding(nn.Module):
    def __init__(self, d_model=64, device='cuda'):
        super(PositionalEncoding, self).__init__()
        self.d_model = d_model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Ensure that the d_model is even
        assert d_model % 2 == 0, "d_model should be an even number"

    def forward(self, mz_values):
        # Prepare a tensor for m/z values
        mz_tensor = torch.tensor(mz_values, device=self.device).unsqueeze(1)  # shape: (n_peaks, 1)
        mz_tensor = mz_tensor.view(-1, 1)
        # Compute the positional encoding
        div_term = torch.exp(torch.arange(0, self.d_model, 2, device=self.device) * (-math.log(10000.0) / self.d_model))
        mz_tensor = mz_tensor.expand(-1, div_term.size(0))  # shape: (n_peaks, d_model//2)
        pe_sin = torch.sin(mz_tensor * div_term)  # shape: (n_peaks, d_model/2)
        pe_cos = torch.cos(mz_tensor * div_term)  # shape: (n_peaks, d_model/2)
        
        # Concatenate sine and cosine encodings
        pe = torch.cat((pe_sin, pe_cos), dim=1)  # shape: (n_peaks, d_model)

        return pe


# Distance matrix function
def dist_matrix(embeddings):
    if embeddings.ndim == 1:
        embeddings = embeddings[:, np.newaxis]  # Reshape to (n, 1)
    diff = embeddings[:, np.newaxis, :] - embeddings[np.newaxis, :, :]
    sq_diff = np.sum(diff ** 2, axis=2)
    distance_matrix = np.sqrt(sq_diff)
    return distance_matrix

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Example parameters
input_dim = 64  # d_model
nhead = 8  # Number of heads in multihead attention
nhid = 256  # Dimension of the feedforward network
nlayers = 6  # Number of transformer layers
d_model = 64  # Embedding size

# Example: Generate random input data (Distance matrix from random embeddings)
file_path = '/home/ninak/tailor.assign-confidence.filtered10.txt'

# Read the file into a DataFrame
df = pd.read_csv(file_path, sep='\t')

# Extract the values from the "spectrum precursor m/z" column to an array
mz_values = df['spectrum precursor m/z'].values
pe_matrix = torch.tensor(mz_values, dtype=torch.float32)

# Create a distance matrix from the embedding matrix
pe_matrix_cpu = pe_matrix.detach().cpu().numpy()
dist_m = dist_matrix(pe_matrix_cpu)

# Convert distance matrix to torch tensor
distance_matrix_tensor = torch.tensor(dist_m, dtype=torch.float32).cuda()
dataloader = DataLoader(distance_matrix_tensor, batch_size=1)

# Transformer model
model = TransformerModel(input_dim, nhead, nhid, nlayers, d_model=d_model, dropout=0.1)
model.to(device)

def train(model, train_loader):
        model.train()
        running_loss = 0.0
        for inputs in train_loader:
            outputs = model(inputs.to('cuda'))
            print(outputs)
            return
            
train(model, dataloader)

# Move model and data to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
