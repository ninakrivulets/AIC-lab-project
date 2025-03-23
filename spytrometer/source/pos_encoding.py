import torch
import math
import torch.nn as nn

class PositionalEncoding(nn.Module):
    def __init__(self, d_model=64, device='cuda'):
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
        length = array.shape[0]  # Expecting an array of at least 1D (150,)
        
        matrix_1 = array.unsqueeze(0).expand(length, -1)  # First matrix with the array in the rows
        matrix_2 = array.unsqueeze(1).expand(-1, length)  # Matrix with the array in the columns
        diff = torch.abs(matrix_1 - matrix_2)  # Distance matrix
        return diff

    def tri_cube(self, array):
        array = torch.tensor(array, dtype=torch.float32, device=self.device)
        array = torch.abs(array / 0.05)
        value = torch.where(array <= 1, (1 - array ** 3) ** 3, torch.tensor(0.0, device=self.device))
        return value

    def forward(self, mz_values):
        mz_values = torch.tensor(mz_values, device=self.device, dtype=torch.float32)  # Convert m/z values to tensor
        mz_tensor = mz_values.unsqueeze(1)  # shape: (150, 1)
        # Calculate distance matrices for m/z values
       
        diff_matrix = self.dist_matrix(mz_values)

        diff_matrices = torch.tensor(diff_matrix)  # shape: (150, 150, 150)

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
        # Compute the positional encoding
        div_term = torch.exp(torch.arange(0, self.d_model, 2, device=self.device) * (-math.log(10000.0) / self.d_model))
        div_term = div_term.unsqueeze(0).unsqueeze(2)  # shape: (1, d_model/2, 1)
        mz_tensor = mz_tensor.unsqueeze(1)  # shape: (150, 1, 1)
        mz_tensor = mz_tensor.repeat(1, div_term.size(1), 1)  # shape: (150, d_model//2=32, 1)

        pe_sin = torch.sin(mz_tensor * div_term)  # shape: (150, d_model/2)
        pe_cos = torch.cos(mz_tensor * div_term)  # shape: (150, d_model/2)

        # Concatenate sine and cosine encodings
        pe = torch.cat((pe_sin, pe_cos), dim=1)  # shape: (150, d_model, 1)
        pe_p = pe.squeeze(2) #shape (150, 64)
        result = torch.matmul(match_matrices, pe_p) #shape (28, 150, 64)

        
        encoding = torch.sum(result, dim=0) #shape (150, 64)
       
        return encoding