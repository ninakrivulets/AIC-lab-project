import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import pandas as pd
import os
import re
import pyopenms as oms
import matplotlib.pyplot as plt
import numpy as np
import random
import string
import uuid
import base64
import logging
# import pickle
# from utils import get_logger, code_backup
# from torch.utils.tensorboard import SummaryWriter

class PeakEncoding(nn.Module):
    def __init__(self, device, d_model=64):
        super(PeakEncoding, self).__init__()
        self.d_model = d_model
        self.device = device

        # Ensure that the d_model is even
        assert d_model % 2 == 0, "d_model should be an even number"

    def forward(self, mz_values):
        # Prepare a tensor for m/z values
        mz_tensor = torch.tensor(mz_values, device=self.device, dtype=torch.float32).unsqueeze(1)  # shape: (n_peaks, 1)
        mz_tensor = mz_tensor.view(-1, 1) # reshaping 
        # Compute the positional encoding
        div_term = torch.exp(torch.arange(0, self.d_model, 2, device=self.device) * (-math.log(10000.0) / self.d_model))
        mz_tensor = mz_tensor.expand(-1, div_term.size(0))  # shape: (n_peaks, d_model//2)
        pe_sin = torch.sin(mz_tensor * div_term)  # shape: (n_peaks, d_model/2)
        pe_cos = torch.cos(mz_tensor * div_term)  # shape: (n_peaks, d_model/2)
        
        # Concatenate sine and cosine encodings
        pe = torch.cat((pe_sin, pe_cos), dim=1)  # shape: (n_peaks, d_model)

        return pe


class PeakEncodingWithDistances(nn.Module):
    def __init__(self, d_model=64, device='cuda'):
        super(PeakEncodingWithDistances, self).__init__()
        self.d_model = d_model
        self.device = device

        assert d_model % 2 == 0, "d_model should be an even number"

        # Amino acid mass dictionary
        self.aa_mass = {
            "G": 57.02146, "A": 71.03711, "S": 87.03203, "P": 97.05276, "V": 99.06841,
            "T": 101.04768, "C": 103.00919, "L": 113.08406, "I": 113.08406, "N": 114.04293,
            "D": 115.02694, "Q": 128.05858, "K": 128.09496, "E": 129.04259, "M": 131.04049,
            "H": 137.05891, "F": 147.06841, "U": 150.95364, "R": 156.10111, "Y": 163.06333,
            "W": 186.07931, "1": 147.035399, "2": 166.998359435, "3": 181.014009505,
            "4": 243.029659575, "5": 170.10552805, "6": 170.11676105, "7": 142.11061305
        }

        # Convert masses to tensor
        self.aa_masses_tensor = torch.tensor(list(self.aa_mass.values()), device=self.device)

    def dist_matrix(self, array):
        #array = torch.tensor(array, device=self.device)
        length = array.shape[0]  # Expecting an array of at least 1D (150,)
        
        matrix_1 = array.unsqueeze(0).expand(length, -1)  # First matrix with the array in the rows
        matrix_2 = array.unsqueeze(1).expand(-1, length)  # Matrix with the array in the columns
        diff = torch.abs(matrix_1 - matrix_2)  # Distance matrix
        return diff

    def tri_cube(self, array):
        #array = torch.tensor(array, dtype=torch.float32, device=self.device)
        array = torch.abs(array / 0.05)
        value = torch.where(array <= 1, (1 - array ** 3) ** 3, torch.tensor(0.0, device=self.device))
        return value

    def forward(self, mz_values):
        mz_values = torch.tensor(mz_values, device=self.device, dtype=torch.float32)  # Convert m/z values to tensor
        mz_tensor = mz_values.unsqueeze(1)  # shape: (150, 1)
        # Calculate distance matrices for m/z values
        #diff_matrices = []
        #for mz_value in mz_values:
        diff_matrices = self.dist_matrix(mz_values)
        #diff_matrices.append(diff_matrix)

        #diff_matrices = torch.tensor(diff_matrix)  # shape: (150, 150, 150)

        # Calculate mass difference matrices for each amino acid mass
        mass_diff_matrices = []
        for mass in self.aa_masses_tensor:
            mass_diff_matrix = torch.abs(diff_matrices - mass)  # shape: (150, 150, 150)
            mass_diff_matrices.append(mass_diff_matrix)  # list of 28 tensors

        # Stack to get a tensor of shape (28, 150, 150, 150)
        mass_diff_matrices = torch.stack(mass_diff_matrices)  # shape: (28, 150, 150)
        #print('Mass diff shape', mass_diff_matrices.shape)
        # Apply tri-cube function to match mass differences
        match_matrices = self.tri_cube(mass_diff_matrices) # shape (28, 150, 150)

        # Create mask for diagonal elements
        diag_mask = torch.eye(match_matrices.shape[1], match_matrices.shape[2], dtype=torch.bool, device=self.device)  # shape: (150, 150)

        # Expand mask to (28, 150, 150)
        diag_mask = diag_mask.unsqueeze(0).expand(match_matrices.shape[0], -1, -1)  # shape: (28, 150, 150)

        # Set diagonal values to 1
        match_matrices[diag_mask] = 1.0 # shape (28, 150, 150)
        #match_matrices = match_matrices.permute(2, 1, 0)  #(150, 150,  28) 
        # Compute the positional encoding
        div_term = torch.exp(torch.arange(0, self.d_model, 2, device=self.device) * (-math.log(10000.0) / self.d_model))
        div_term = div_term.unsqueeze(0).unsqueeze(2)  # shape: (1, d_model/2, 1)
        #mz_tensor = mz_tensor.expand(-1, div_term.size(0), -1)  # shape: (150, d_model//2, 1)
        mz_tensor = mz_tensor.unsqueeze(1)  # shape: (150, 1, 1)
        mz_tensor = mz_tensor.repeat(1, div_term.size(1), 1)  # shape: (150, d_model//2=32, 1)

        pe_sin = torch.sin(mz_tensor * div_term)  # shape: (150, d_model/2)
        pe_cos = torch.cos(mz_tensor * div_term)  # shape: (150, d_model/2)

        # Concatenate sine and cosine encodings
        pe = torch.cat((pe_sin, pe_cos), dim=1)  # shape: (150, d_model, 1)
        pe_p = pe.squeeze(2) #shape (150, 64)
        result = torch.matmul(match_matrices, pe_p) #shape (28, 150, 64)
        encoding = torch.sum(result, dim=0)

        return encoding


class AAEmbedding(nn.Module):
    def __init__(self, device, embedding_dim=64):
        super(AAEmbedding, self).__init__()
        self.embedding_dim = embedding_dim
        self.device = device
        # creating a dictionary to map amino acids to indices
        self.amino_acids = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 
                            'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y',
                            'X', 'O','U']

        self.aa_to_idx = {aa: idx for idx, aa in enumerate(self.amino_acids)}
        
        # an embedding layer
        self.embedding_layer = nn.Embedding(num_embeddings=len(self.amino_acids), embedding_dim=self.embedding_dim).to(device)
        window = 5
        self.proteome_kernel = nn.Conv1d(in_channels=self.embedding_dim, out_channels=self.embedding_dim, kernel_size=window*2+1, padding=window, stride=window).to(device)

    def forward(self, sequence):
        # Convert sequence to indices
        sequence_indices = torch.tensor([self.aa_to_idx[aa] for aa in sequence], device=self.device)
        # Pass the indices through the embedding layer
        embedded_sequence = self.embedding_layer(sequence_indices).unsqueeze(0).transpose(1,2)
        # print(embedded_sequence.shape)
        embedded_sequence = self.proteome_kernel(embedded_sequence).transpose(1,2)
        return embedded_sequence # Adding batch dimension
    

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads 
        self.d_model = d_model
        self.depth = d_model // num_heads 
        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)
        self.dense = nn.Linear(d_model, d_model)

    def split_heads(self, x, batch_size):
        # reshapeing x into a tensor of shape (batch_size, seq_length, num_heads, depth)
        x = x.view(batch_size, -1, self.num_heads, self.depth)
        # changeing the order of dimensions to (batch_size, num_heads, seq_length, depth)
        return x.permute(0, 2, 1, 3)

    def forward(self, q, k, v):
        batch_size = q.size(0)

        q = self.split_heads(self.wq(q), batch_size)
        k = self.split_heads(self.wk(k), batch_size)
        v = self.split_heads(self.wv(v), batch_size)

        attention_scores = F.softmax(torch.matmul(q, k.transpose(-2, -1)) / (self.depth ** 0.5), dim=-1)
        attention_output = torch.matmul(attention_scores, v)

        attention_output = attention_output.permute(0, 2, 1, 3).contiguous().view(batch_size, -1, self.d_model)
        # permute(0, 2, 1, 3) changes dimensions so:
         #0- axis (batch_size) stays,
         #2-axis (seq_length) becomes 1,
         #1-axis (num_heads) becomes 2,
         # 3-я axis (depth) stays.
         # we get tensor with shape (batch_size, seq_len_q, num_heads, depth).
        return self.dense(attention_output)

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff=2048):
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        return self.linear2(F.relu(self.linear1(x)))

class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads):
        super(DecoderLayer, self).__init__()
        # self.mha1 = MultiHeadAttention(d_model, num_heads)
        self.conv1d = nn.Conv1d(in_channels=d_model, out_channels=d_model, kernel_size=5, padding=2)
        self.mha2 = MultiHeadAttention(d_model, num_heads)
        self.ffn = FeedForward(d_model)
        self.layernorm1 = nn.LayerNorm(d_model)
        self.layernorm2 = nn.LayerNorm(d_model)
        self.layernorm3 = nn.LayerNorm(d_model)

    def forward(self, x, enc_output):
        # attn1 = self.mha1(x, x, x)
        # print(x.shape) # is [1, 24, 64]
        x = x.permute(0, 2, 1)  # Change shape from (batch, seq_len, d_model) to (batch, d_model, seq_len)
        # print(x.shape) # is ([1, 64, 24])
        conv_output = self.conv1d(x)
        # print(conv_output.shape) # is [1, 64, 24]
        conv_output = conv_output.permute(0, 2, 1)  # Change shape back to (batch, seq_len, d_model)
        x = x.permute(0, 2, 1) # shape [1, 24, 64]
        x = self.layernorm1(x + conv_output)  # Residual connection
        attn2 = self.mha2(x, enc_output, enc_output)
        x = self.layernorm2(x + attn2)  # Residual connection
        ffn_output = self.ffn(x)
        return self.layernorm3(x + ffn_output)  # Residual connection

class Decoder(nn.Module):
    def __init__(self, d_model, num_heads, num_layers):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList([DecoderLayer(d_model, num_heads) for _ in range(num_layers)])

    def forward(self, x, enc_output):
        for layer in self.layers:
            x = layer(x, enc_output)
        return x

class Transformer(nn.Module):
    def __init__(self, device, d_model=64, num_heads=8, num_layers=6):
        super(Transformer, self).__init__()
        self.encoder_positional_encoding = PeakEncodingWithDistances(d_model, device)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads).to(device)
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers).to(device)
        
        self.decoder_embedding = AAEmbedding(device, embedding_dim=d_model)  # embedding_dim is 64
        self.decoder = Decoder(d_model, num_heads, num_layers).to(device)
        
        self.final_layer = nn.Linear(d_model, 1).to(device)  # Output size is 1 for the center prediction
        self.device = device

    def forward(self, encoder_input, decoder_input):
        encoder_input = self.encoder_positional_encoding(encoder_input)
        enc_output = self.encoder(encoder_input)

        decoder_input_embedded = self.decoder_embedding(decoder_input)
        # print(f"decoder input (embedded) shape: {decoder_input_embedded.shape}")
        # print(f"encoder ouput shape: {enc_output.shape}")
        dec_output = self.decoder(decoder_input_embedded, enc_output)
        # print(f"encoder ouput shape: {dec_output.shape}")

        final_output = self.final_layer(dec_output).squeeze(-1)
        
        # Applying softmax to get the distribution
        # distribution = F.softmax(final_output.squeeze(-1), dim=-1)

        return final_output
    
def get_logger(exp_id, log_path):
    logger = logging.getLogger('main')
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        file_handler = logging.FileHandler(f"{log_path}/experiment_{exp_id}.log")
        file_handler.setLevel(logging.INFO)
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s [%(filename)s:%(lineno)d %(funcName)s]'
        )

        #formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

    return logger

def generate_uuid():
    """Generates a uuid 4 string, in this context for tracking each run of the experiment
    Returns:
        an ascii friendly uuid4 string.
    """
    return base64.urlsafe_b64encode(uuid.uuid4().bytes).rstrip(b"=").decode("ascii")

def remove_brackets(input_string):
    return re.sub(r'\[.*?\]', '', input_string)

def add_random_letters(input_string):
    amino_acids = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 
                            'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
    # Generate 100 random capital letters
    random_letters_front = ''.join(random.choices(amino_acids, k=100))
    random_letters_back = ''.join(random.choices(amino_acids, k=100))
    # Add random letters before and after the input string
    return random_letters_front + input_string + random_letters_back