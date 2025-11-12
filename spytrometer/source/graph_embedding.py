import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv

class PeakEncodingWithDistances_Graph(nn.Module):
    def __init__(self, d_model=64, device='cuda'):
        super().__init__()
        self.d_model = d_model
        self.device = device
        assert d_model % 2 == 0

        self.aa_mass = {
            "G": 57.02146, "A": 71.03711, "S": 87.03203, "P": 97.05276, "V": 99.06841,
            "T": 101.04768, "C": 103.00919, "L": 113.08406, "I": 113.08406, "N": 114.04293,
            "D": 115.02694, "Q": 128.05858, "K": 128.09496, "E": 129.04259, "M": 131.04049,
            "H": 137.05891, "F": 147.06841, "U": 150.95364, "R": 156.10111, "Y": 163.06333,
            "W": 186.07931, "1": 147.035399, "2": 166.998359435, "3": 181.014009505,
            "4": 243.029659575, "5": 170.10552805, "6": 170.11676105, "7": 142.11061305
        }
        self.aa_masses_tensor = torch.tensor(list(self.aa_mass.values()), device=self.device)

    def dist_matrix(self, arr):
        arr = arr.unsqueeze(0)
        diff = torch.abs(arr - arr.T)
        return diff

    def tri_cube(self, array):
        array = torch.abs(array / 0.05)
        return torch.where(array <= 1, (1 - array**3)**3, torch.tensor(0.0, device=array.device))

    def forward(self, mz_values):
        mz_values = torch.tensor(mz_values, device=self.device)
        diff = self.dist_matrix(mz_values)  # (N, N)

        mass_diff_matrices = torch.abs(diff.unsqueeze(0) - self.aa_masses_tensor[:, None, None])
        #match_matrices = self.tri_cube(mass_diff_matrices)
        # OR: sum over amino acids to get final adjacency weights
        adjacency = mass_diff_matrices #torch.sum(mass_diff_matrices, dim=0)  # (N, N)
        print(adjacency.shape)
        # make diagonal = 1 (self loops)
        #adjacency.fill_diagonal_(0.0)

        # build positional encodings
        N = mz_values.shape[0]
        div_term = torch.exp(torch.arange(0, self.d_model, 2, device=self.device) * (-math.log(10000.0)/self.d_model))
        div_term = div_term.unsqueeze(0)  # (1, d_model/2)

        mz_expand = mz_values.unsqueeze(1)  # (N,1)
        pe_sin = torch.sin(mz_expand * div_term)
        pe_cos = torch.cos(mz_expand * div_term)
        pe = torch.cat((pe_sin, pe_cos), dim=1)  # (N, d_model)

        return pe, adjacency

def build_graph_from_match_matrices(pe_embeddings, match_matrices, threshold=0.02):
    """
    match_matrices: (28, N, N)
    pe_embeddings:  (N, d_model)
    """
    num_aa, N, _ = match_matrices.shape

    edge_indices = []
    edge_attrs = []
    aa_mass = {
            "G": 57.02146, "A": 71.03711, "S": 87.03203, "P": 97.05276, "V": 99.06841,
            "T": 101.04768, "C": 103.00919, "L": 113.08406, "I": 113.08406, "N": 114.04293,
            "D": 115.02694, "Q": 128.05858, "K": 128.09496, "E": 129.04259, "M": 131.04049,
            "H": 137.05891, "F": 147.06841, "U": 150.95364, "R": 156.10111, "Y": 163.06333,
            "W": 186.07931, "1": 147.035399, "2": 166.998359435, "3": 181.014009505,
            "4": 243.029659575, "5": 170.10552805, "6": 170.11676105, "7": 142.11061305
        }
    keys_list = list(aa_mass.keys())

    # iterate over amino acids
    for aa_idx in range(num_aa):
        mat = match_matrices[aa_idx] # (N, N)
        #print("SINGLE", mat)
        mask = mat < threshold                 
        coords = mask.nonzero(as_tuple=False) # (E, 2)
        #print('COORDS', coords)
        if coords.shape[0] > 0:
            # store edge_index for this aa
            edge_indices.append(coords.T) # (2, E)

            # mass value as weight
            vals = mat[mask] # (E,)
            
            # add aa_index to identify which amino acid caused this edge
            aa_feat = torch.full((vals.shape[0], 1), aa_idx, device=vals.device)
            
            # print('AAF', aa_feat)
            # edge attr: [amino acid weight, amino-acid index]
            key = keys_list[aa_idx]
            #vals = torch.tensor(aa_mass[key], device=aa_feat.device)
            vals = torch.full((vals.shape[0], 1), aa_mass[key], device=aa_feat.device)
            #print('VALS', vals)
            ea = torch.cat((vals, aa_feat), dim=1)
            edge_attrs.append(ea)

    # Concatenate across amino acids
    edge_index = torch.cat(edge_indices, dim=1) # (2, total_edges)
    edge_attr = torch.cat(edge_attrs, dim=0) # (total_edges, 2)

    data = Data(
        x=pe_embeddings, # node features
        edge_index=edge_index, # edges
        edge_attr=edge_attr # weights and aa-id
    )
    return data


# Example GNN model 
class GNN(nn.Module):
    def __init__(self, in_dim, hidden=64, out_dim=64):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden)
        self.conv2 = GCNConv(hidden, out_dim)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = torch.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return x

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# fake peaks
mz_values = torch.tensor([10, 67.021461, 164.074221, 350.153531]) 

encoder = PeakEncodingWithDistances_Graph(d_model=64, device=device)
pe, adjacency = encoder(mz_values)
print(adjacency)
print(adjacency < 0.02)
data = build_graph_from_match_matrices(pe, adjacency, threshold=0.01)
data = data.to(device)

model = GNN(in_dim=64).to(device)
embeddings = model(data) # (N, 64)

print("Node embeddings shape:", embeddings.shape)
print("Edge count:", data.edge_index.shape[1])
print(embeddings.shape)
print(data.edge_index)

print(data.edge_attr)
