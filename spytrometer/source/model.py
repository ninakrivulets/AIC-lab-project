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
        mz_tensor = mz_tensor.view(-1, 1) # reshaping 
        # Compute the positional encoding
        div_term = torch.exp(torch.arange(0, self.d_model, 2, device=self.device) * (-math.log(10000.0) / self.d_model))
        mz_tensor = mz_tensor.expand(-1, div_term.size(0))  # shape: (n_peaks, d_model//2)
        pe_sin = torch.sin(mz_tensor * div_term)  # shape: (n_peaks, d_model/2)
        pe_cos = torch.cos(mz_tensor * div_term)  # shape: (n_peaks, d_model/2)
        
        # Concatenate sine and cosine encodings
        pe = torch.cat((pe_sin, pe_cos), dim=1)  # shape: (n_peaks, d_model)

        return pe


class Embedding(nn.Module):
    def __init__(self, embedding_dim=64, device='cuda'):
        super(Embedding, self).__init__()
        self.embedding_dim = embedding_dim
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # creating a dictionary to map amino acids to indices
        self.amino_acids = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 
                            'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
        self.aa_to_idx = {aa: idx for idx, aa in enumerate(self.amino_acids)}
        
        # an embedding layer
        self.embedding_layer = nn.Embedding(num_embeddings=len(self.amino_acids), embedding_dim=self.embedding_dim)

    def forward(self, sequence):
        # Convert sequence to indices
        sequence_indices = torch.tensor([self.aa_to_idx[aa] for aa in sequence], device=self.device)
        # Pass the indices through the embedding layer
        embedded_sequence = self.embedding_layer(sequence_indices)
        return embedded_sequence.unsqueeze(0)  # Adding batch dimension
    

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
         #0-ая ось (batch_size) stays,
         #2-ая ось (seq_length) становится 1-ой,
         #1-ая ось (num_heads) становится 2-ой,
         # 3-я ось (depth) остается на месте.
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
        self.mha1 = MultiHeadAttention(d_model, num_heads)
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
    def __init__(self, d_model=64, num_heads=8, num_layers=6, device='cuda'):
        super(Transformer, self).__init__()
        self.encoder_positional_encoding = PositionalEncoding(d_model)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads)
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        
        self.decoder_embedding = Embedding()  # embedding_dim is 64
        self.decoder = Decoder(d_model, num_heads, num_layers)
        
        self.final_layer = nn.Linear(d_model, 1)  # Output size is 1 for the center prediction
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    def forward(self, encoder_input, decoder_input):
        encoder_input = self.encoder_positional_encoding(encoder_input)
        enc_output = self.encoder(encoder_input)

        decoder_input_embedded = self.decoder_embedding(decoder_input)
        print(f"decoder input (embedded) shape: {decoder_input_embedded.shape}")
        print(f"encoder ouput shape: {enc_output.shape}")
        dec_output = self.decoder(decoder_input_embedded, enc_output)

        final_output = self.final_layer(dec_output)
        
        # Applying softmax to get the distribution
        distribution = F.softmax(final_output.squeeze(-1), dim=-1)

        return distribution
    
def get_logger(exp_id, log_path):
    #logger = logging.getLogger(f"Experiment_{exp_id}")
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
