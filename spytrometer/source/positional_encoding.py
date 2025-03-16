
import torch
import math
import numpy as np
import pandas as pd
import torch.nn as nn
import os
import pickle

class PositionalEncoding(nn.Module):
    def __init__(self, device, d_model=64):
        super(PositionalEncoding, self).__init__()
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
        array = torch.tensor(array, device=self.device)
        length = array.shape[0]
        matrix_1 = array.unsqueeze(0).expand(length, -1)  # first matrix with the array in the rows
        matrix_2 = array.unsqueeze(1).expand(-1, length)  # matrix with the array in the columns
        diff = torch.abs(matrix_1 - matrix_2)  # distance matrix
        return diff
    
    def tri_cube(self, array):
        array = torch.tensor(array, dtype=torch.float32, device=self.device)
        array = torch.abs(array / 0.05)
        value = torch.where(array <= 1, (1 - array ** 3) ** 3, torch.tensor(0.0, device=self.device))
        return value

    def forward(self, mz_values):
        mz_values = torch.tensor(mz_values, device=self.device, dtype=torch.float32) # Convert m/z values to tensor
        #print('Tensor shape', mz_values.shape)
        mz_tensor = torch.tensor(mz_values, device=self.device, dtype=torch.float32).unsqueeze(1)  # shape: (n_peaks, 1)
        
        #print('next shape', mz_tensor.shape)
        diff_matrices = []
        for mz_value in list(mz_values):
            diff_matrix = self.dist_matrix(mz_value)
            diff_matrices.append(diff_matrix)

        diff_matrices = torch.stack(diff_matrices)  # shape: (n_spectra, n_peaks, n_peaks)

        mass_diff_matrices = []
        for mass in self.aa_masses_tensor:
            # Calculate difference between each element of diff_matrix and amino acid mass
            mass_diff_matrix = torch.abs(diff_matrices - mass)  # shape: (n_spectra, n_peaks, n_peaks)
            mass_diff_matrices.append(mass_diff_matrix)  # we get a list of 28 tensors 
        
        # stack 28 matrices: shape (28, n_spectra, n_peaks, n_peaks)
        mass_diff_matrices = torch.stack(mass_diff_matrices)  # shape: (28, n_spectra, n_peaks, n_peaks)
        
        #print("\nMass Difference Matrices (28 matrices):\n", mass_diff_matrices[0])
        # Check if distances match any amino acid mass

        match_matrices = self.tri_cube(mass_diff_matrices)
        
        # Compute the positional encoding
        div_term = torch.exp(torch.arange(0, self.d_model, 2, device=self.device) * (-math.log(10000.0) / self.d_model))
        div_term = div_term.unsqueeze(0).unsqueeze(2)  # shape: (1, d_model/2, 1)
        mz_tensor = mz_tensor.expand(-1, div_term.size(0), -1)  # shape: (n_peaks, d_model//2, 150)
        
        pe_sin = torch.sin(mz_tensor * div_term)  # shape: (n_peaks, d_model/2)
        pe_cos = torch.cos(mz_tensor * div_term)  # shape: (n_peaks, d_model/2)
        
        # Concatenate sine and cosine encodings
        pe = torch.cat((pe_sin, pe_cos), dim=1)  # shape: (n_peaks, d_model)
        #print('PE shape', pe.shape)

        match_encoding = torch.zeros(1, 1000, 150, 64) # torch.zeros(1000, 64, 150)
        for i in range(150):
            for k in range(28):  # Looping over the batch dimension of match_matrices
                a_ij = match_matrices[k, :, i, :]  # Extracts the row i of the 150x150 matrix
                p_i = pe[:, :, i]  # Extracts the i-th element of the positional encoding
                # p_i shape: [1000, 64]
                # Reshape a_ij to [1000, 150, 1] so it can be broadcasted
                a_ij_exp = a_ij.unsqueeze(2)  # Now a_ij_exp has shape [1000, 150, 1]
                #print('AIJ SHAPE', a_ij_exp.shape)
                # p_i_exp should be reshaped to [1000, 1, 150] for broadcasting
                p_i_exp = p_i.unsqueeze(2)  # Now p_i_exp has shape [1000, 64, 1]
                
                # The pe_exp has shape [1000, 64, 150], we need to make sure it aligns for addition
                # Now, compute the element-wise addition of the positional encodings (p_i + p_j)
                pe_exp = pe[:, :, :]  # pe_matrix has shape [1000, 64, 150]
                # p_i_exp has shape [1000, 1, 150], and pe_exp has shape [1000, 64, 150]
                # We will broadcast p_i_exp to [1000, 64, 150] for the addition
                p_plus_pe = p_i_exp + pe_exp  # Now p_plus_pe has shape [1000, 64, 150]
                p_plus_pe = p_plus_pe.permute(0, 2, 1)  
                # Now we compute the element-wise multiplication and sum over j
                # We need to sum over j, so we need to expand the dimensions appropriately:
                add = a_ij_exp * (p_plus_pe)
                add = add.unsqueeze(0).to(device)
                #print("ADD", add.shape)
                match_encoding = match_encoding.to(device)
                match_encoding += torch.sum(add, dim=0)
        #print('SHAPE', match_encoding.shape)
        match_encoding = match_encoding.squeeze(0)
        # Reshape match_encoding from (1000, 150, 64) to (150000, 64)
        match_encoding = match_encoding.view(-1, 64)  

        print('SHAPE', match_encoding.shape)
        return match_encoding
    

pickle_file_path = "/blob/dda/PXD028806/training_data/PXD028806_tailor_11.pkl"
            
with open(pickle_file_path, "rb") as f:
    data = pickle.load(f)
mz_values = data["mz_values"]
print('Shape:', mz_values.shape)
exp_dir = '/home/ninak/exps'
if not os.path.exists(exp_dir):
    os.makedirs(exp_dir)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

pos_encoder = PositionalEncoding(device)
pe = pos_encoder(mz_values)
print(pe)