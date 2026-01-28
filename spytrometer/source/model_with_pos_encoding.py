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

# from torch_geometric.data import Data
# from torch_geometric.nn import GCNConv


AA_MASS = {
    "A": 71.03711,
    "R": 156.10111,
    "N": 114.04293,
    "D": 115.02694,
    "C": 103.00919,
    "E": 129.04259,
    "Q": 128.05858,
    "G": 57.02146,
    "H": 137.05891,
    "I": 113.08406,
    "L": 113.08406,
    "K": 128.09496,
    "M": 131.04049,
    "F": 147.06841,
    "P": 97.05276,
    "S": 87.03203,
    "T": 101.04768,
    "W": 186.07931,
    "Y": 163.06333,
    "V": 99.06841
}

PROTON = 1.007276466812
Nterm = 1.007825035

import re

def parse_modified_sequence(sequence: str):
    """Parse peptide sequence with inline PTMs like M[15.99]
    """
    pattern = re.compile(r"([A-Z])(?:\[([+-]?\d*\.?\d+)\])?")
    residues = []

    for aa, mod in pattern.findall(sequence):
        if aa not in AA_MASS:
            raise ValueError(f"Unknown amino acid: {aa}")

        mass = AA_MASS[aa]
        if mod:
            mass += float(mod)

        residues.append(mass)

    return residues

def compute_b_ions_modified(sequence: str):
   
    residue_masses = parse_modified_sequence(sequence)

    b_ions = []
    cumulative_mass = Nterm

    for i, mass in enumerate(residue_masses, start=1):
        cumulative_mass += mass
        b_ions.append((cumulative_mass))

    return b_ions


# def compute_b_ions(sequence: str):
    
#     b_ions = []
#     cumulative_mass = 0.0

#     for i, aa in enumerate(sequence, start=1):
#         if aa not in AA_MASS:
#             raise ValueError(f"Unknown amino acid: {aa} in {sequence}")

#         cumulative_mass += AA_MASS[aa]
#         b_mass = cumulative_mass + PROTON
#         b_ions.append((b_mass))

#     return b_ions

class PeakEncoding(nn.Module):
    def __init__(self, d_model, device):
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

 #Spectrum encoding with binary vector indicating peak distances. if a distance between two peaks is equal to a given amino acid mass
class SpectrumEncoding(nn.Module): 
    def __init__(self, d_model=64, device='cuda'):
        super(SpectrumEncoding, self).__init__()
        self.d_model = d_model
        self.device = device
        AA_pair_mass = {}
        for key, value in AA_MASS.items():
            AA_pair_mass[key] = value
        for key1, value1 in AA_MASS.items():
            for key2, value2 in AA_MASS.items():
                AA_pair_mass[key1+key2] = value1+value2

        self.masses = torch.tensor(list(AA_pair_mass.values()), dtype=torch.float32, device=device)
        self.AA_num = self.masses.shape[0] # Number of Amino Acids
        # print(AA_pair_mass)
        # print(self.masses)
        # exit()

        scale = torch.tensor(0.001, dtype=torch.float32, device=device)
        base =  torch.tensor(5000.0, dtype=torch.float32, device=device)

        # The spectrum peak embedding will have a dimension  = peak_pos_embedding + AA_num = d_model
        peak_pos_embeddig_dim = d_model-self.AA_num    # d_model should be larger than AA_num, 
        if peak_pos_embeddig_dim < 1:
            print("Fatal, model_dim is not big enough")
            exit()

        exp_term = torch.arange(1, peak_pos_embeddig_dim+1, 2, dtype=torch.float32, device=self.device)*2/peak_pos_embeddig_dim
        self.div_term =1.0/(scale * torch.pow(base, exp_term))      

    def forward(self, mz_peaks): 
        #mz_peaks = torch.tensor(peaks_mz, dtype=torch.float32, device=device) of shape (150,), one dimensional, 

        # Get the peak embeddings, with sin-cos
        outer = torch.outer(mz_peaks, self.div_term) # shape: (n_peaks, (d_model- AA_num)/2)
        pe_sin = torch.sin(outer)  # shape: (n_peaks, (d_model- AA_num)/2)
        pe_cos = torch.cos(outer)  # shape: (n_peaks, (d_model- AA_num)/2)
        
        # Concatenate sine and cosine encodings
        spectrum_tensor = torch.cat((pe_sin, pe_cos), dim=1)  # shape: (n_peaks, d_model-AA_num)

        # Get peak pairs whose distances is equal to the mass of one of the amino acids        
        #mz_peaks = torch.tensor(peaks_mz, dtype=torch.float32, device=device) of shape (150,), one dimensional, 

        # diff_AA_idx[i,j,k] indicates that if the peak_i and peak_j has a difference of the mass of amino acid k
        diff_AA_idx = self.peak_distances(mz_peaks)  # shape [peak_num, peak_num, AA_num], 

        spectrum_tensor_with_aa_dist_idx = []        
        diff_AA_cnt = diff_AA_idx.sum(dim=2)   # Keep peak pairs whose distance is equal to one of the amino acid masses; discard the others
        peak_pair_idx = diff_AA_cnt.nonzero()
        for pair in peak_pair_idx:
            if (pair[0] < pair[1]):
                spectrum_tensor_with_aa_dist_idx.append(torch.cat((spectrum_tensor[pair[0],:], diff_AA_idx[pair[0], pair[1], :])))

        out = torch.stack(spectrum_tensor_with_aa_dist_idx)   # shape: (peaks, d_model), 2-D tensor
        return out

        # peak_dist_mx = self.peak_distances(mz_peaks)

    def peak_distances(self, mz_peaks):
        # mz_peaks = torch.tensor(peaks_mz, dtype=torch.float32, device=device) of shape (peak_num,), one dimensional, max peak num is around 150

        peaks_num = mz_peaks.shape[0]     # Expecting an 1-D array of values. Shape: [peak_num]
        peaks_vector = mz_peaks.unsqueeze(1)  # we make each value as a 1-D vector, shape: [peak_num, 1]
                
        diff = torch.cdist(peaks_vector, peaks_vector)   # shape: [peak_num, peak_num], 2-D tensor, [i,j] contains  | peak_i - peak_j |
        diff_AA = diff.unsqueeze(2).repeat(1,1,self.AA_num)  # shape: [peak_num, peak_num, AA_num], 3-D tensor

        #self.masses list of AA masses: shape [AA_num]
        AA_mass = self.masses.unsqueeze(0).repeat(peaks_num, 1)  # shape [peak_num, AA_num]
        AA_mass = AA_mass.unsqueeze(0).repeat(peaks_num, 1, 1)   # shape [peak_num, peak_num, AA_num]

        diff_AA_mass = torch.abs(AA_mass - diff_AA)   # shape [peak_num, peak_num, AA_num]

        # diff_AA_idx[i,j,k] indicates that if the peak_i and peak_j has a difference of the mass of amino acid k
        diff_AA_idx = torch.where( diff_AA_mass < 0.02, 1.0, 0.0)   # shape [peak_num, peak_num, AA_num], 
        
        return diff_AA_idx
    

class AAEmbedding(nn.Module):
    def __init__(self, device, embedding_dim=64, kernel_stride=7):
        super(AAEmbedding, self).__init__()
        self.embedding_dim = embedding_dim
        self.device = device
        # creating a dictionary to map amino acids to indices
        self.amino_acids = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 
                            'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y',
                            'X', 'O','U', '1', '2', '3', '4', '5', '6', '7']

        self.aa_to_idx = {aa: idx for idx, aa in enumerate(self.amino_acids)}
        
        # an embedding layer
        self.embedding_layer = nn.Embedding(num_embeddings=len(self.amino_acids), embedding_dim=self.embedding_dim).to(device)
        
        self.proteome_kernel = nn.Conv1d(in_channels=self.embedding_dim, out_channels=self.embedding_dim, kernel_size=kernel_stride*2+1, padding=kernel_stride, stride=kernel_stride).to(device)

    def forward(self, sequence):
        # Convert sequence to indices
        sequence_indices = torch.tensor([self.aa_to_idx[aa] for aa in sequence], device=self.device)
        # Pass the indices through the embedding layer
        embedded_sequence = self.embedding_layer(sequence_indices).unsqueeze(0)#.transpose(1,2)
        # print("embedded_sequence.shape", embedded_sequence.shape)
        # embedded_sequence = self.proteome_kernel(embedded_sequence).transpose(1,2)
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
        # print(x.shape) # is [1, 24, 64], dim:64, seq_len: 24
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
    def __init__(self, device, d_model=64, num_heads=8, num_layers=6, kernel_stride=7):
        super(Transformer, self).__init__()
        
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads).to(device)
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers).to(device)
        
        self.decoder_embedding = AAEmbedding(device, embedding_dim=d_model, kernel_stride=kernel_stride)  # embedding_dim is 64
        self.decoder = Decoder(d_model, num_heads, num_layers).to(device)

        self.final_layer = nn.Conv1d(in_channels=d_model, out_channels=1, kernel_size=7, padding=3)

        
        # self.final_layer = nn.Linear(d_model, 1).to(device)  # Output size is 1 for the center prediction
        self.device = device

    def forward(self, spectrum_embedding, decoder_input):#, intensities, decoder_input):
        # spectrum_embedding embedding of the spectrum, and input to the transformer encoder. shape: [peak_num, model dim]
        # decoder_input proteome ( string of amino acid symbols)
        enc_output = self.encoder(spectrum_embedding)
        # print("enc_output:", enc_output.shape)

        decoder_input_embedded = self.decoder_embedding(decoder_input)
        # print(f"decoder input (embedded) shape: {decoder_input_embedded.shape}")
        # print(f"encoder ouput shape: {enc_output.shape}")
        dec_output = self.decoder(decoder_input_embedded, enc_output)
        # print(f"encoder ouput shape: {dec_output.shape}")
        dec_output = dec_output.permute(0, 2, 1)  # Change shape from (batch, seq_len, d_model) to (batch, d_model, seq_len)

        final_output = self.final_layer(dec_output).permute(0, 2, 1).squeeze(-1)
        
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