import torch
import torch.nn as nn

class PeakEncodingWithProteinSeq(nn.Module):
    def __init__(self, d_model = 64, device='cuda'):
        super().__init__()
        self.aa_mass = {
    "G": 57.02146, "A": 71.03711, "S": 87.03203, "P": 97.05276, "V": 99.06841,
    "T": 101.04768, "C": 103.00919, "L": 113.08406, "I": 113.08406, "N": 114.04293,
    "D": 115.02694, "Q": 128.05858, "K": 128.09496, "E": 129.04259, "M": 131.04049,
    "H": 137.05891, "F": 147.06841, "U": 150.95364, "R": 156.10111, "Y": 163.06333,
    "W": 186.07931, "1": 147.035399, "2": 166.998359435, "3": 181.014009505,
    "4": 243.029659575, "5": 170.10552805, "6": 170.11676105, "7": 142.11061305
}
        self.d_model = d_model
        self.device = device

    def forward(self, mz_values, protein_seq):
        """
        mz_values: 1D array-like, shape (N_peaks,)
        protein_seq: str, sequence of amino acid one-letter codes, len==seq_len
        """
        mz_values = torch.tensor(mz_values, dtype=torch.float32, device=self.device)
        N = mz_values.shape[0]
        seq_len = len(protein_seq)
        epsilon = 0.02
        
        # Get masses for the amino acids in protein_seq: (seq_len,)
        mass_list = [self.aa_mass[aa] for aa in protein_seq]
        mass_tensor = torch.tensor(mass_list, dtype=torch.float32, device=self.device) # (seq_len,)

        # Compute all pairwise distances |mz_i - mz_j|, shape (N, N)
        diff_matrix = torch.abs(mz_values.unsqueeze(0) - mz_values.unsqueeze(1))  # (N, N)

        # for every protein position i get mask: (N, N), where 1 if dist == mass[i]+-0.02
        # mass_tensor: (seq_len, 1, 1)
        mass_targets = mass_tensor[:, None, None] # (seq_len, 1, 1)
        diff_matrix_expanded = diff_matrix[None, :, :] # (1, N, N)
        match_mask = (torch.abs(diff_matrix_expanded - mass_targets) <= epsilon) # (seq_len, N, N), bool
        # for every peak, if it appears in at least one pair, mark it 1.
        # for all positions (seq_len) and all peaks (N):
        out = torch.zeros((N, seq_len), dtype=torch.float32, device=self.device)
        # look at 'match by column or by row for every peak
        for seq_idx in range(seq_len):
            mask = match_mask[seq_idx] # (N, N), equals 0 or 1
            # by row, peak1 as mz_i:  riw has 1 if this pek matches as 'i'
            # by column peak2 as mz_j: column has 1 if this peak matched as 'j'
            
            match_peaks = mask.any(dim=0) | mask.any(dim=1) # shape (N,)
            out[:, seq_idx] = match_peaks.float()
        return out # (N_peaks, seq_len) : 0/1


device = 'cuda' 

encoder = PeakEncodingWithProteinSeq(device=device)
mz_values = [99.1, 123.84, 100.2, 18.3, 31, 66.82, 72.12, 199.268]  
protein_seq = 'MVEPTGDV' 
encoded = encoder(mz_values, protein_seq)
print(encoded)