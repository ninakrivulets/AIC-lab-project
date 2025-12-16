import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import logging
import re
import uuid
import base64
import random

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
        # arr: (N,)
        arr = arr.unsqueeze(0)
        diff = torch.abs(arr - arr.T)
        return diff

    def tri_cube(self, array):
        # array may be (28, N, N) or (N,N)
        x = torch.abs(array / 0.05)
        return torch.where(x <= 1, (1 - x ** 3) ** 3, torch.tensor(0.0, device=array.device))

    def forward(self, mz_values):
        # mz_values: 1D iterable or tensor of length N
        mz_values = mz_values.clone().detach()
        #mz_values = torch.tensor(mz_values, device=self.device, dtype=torch.float32)
        N = mz_values.shape[0]

        diff = self.dist_matrix(mz_values)  # (N, N)

        # mass difference matrices per amino acid: shape (28, N, N)
        mass_diffs = torch.abs(diff.unsqueeze(0) - self.aa_masses_tensor[:, None, None])

        #match_matrices = self.tri_cube(mass_diffs)  # (28, N, N)
        match_matrices = (mass_diffs)
        # set diagonal per-channel to 1.0 to indicate self-match (optional)
        diag_mask = torch.eye(N, dtype=torch.bool, device=self.device)
        match_matrices[:, diag_mask] = 0.0

        # positional encoding (sin/cos, standard ) for peaks -> (N, d_model)
        div_term = torch.exp(torch.arange(0, self.d_model, 2, device=self.device) * (-math.log(10000.0) / self.d_model))
        div_term = div_term.unsqueeze(0) # (1, d_model/2)
        mz_expand = mz_values.unsqueeze(1) # (N,1)
        pe_sin = torch.sin(mz_expand * div_term)
        pe_cos = torch.cos(mz_expand * div_term)
        pe = torch.cat((pe_sin, pe_cos), dim=1) # (N, d_model)

        return pe, match_matrices

def build_graph_fast(pe_embeddings, match_matrices, threshold=0.02):
    """
    Builds graph with:
      - node features: pe_embeddings (N, d_model)
      - edges: where match_matrices[a, i, j] < threshold
      - edge_attr: 28-dim binary vector indicating which amino acid matched
    """
    device = match_matrices.device
    num_aa, N, _ = match_matrices.shape  # num_aa = 28

    # mask: (28, N, N) boolean — True if diff matches aa mass
    mask = match_matrices < threshold

    # vectorized extraction of edge coordinates
    aa_idx, i, j = mask.nonzero(as_tuple=True)   # all (aa, i, j) matches

    # (2, E) edge index
    edge_index = torch.stack([i, j], dim=0)

    # --- Build binary indicator matrix ---
    # Every edge must have a 28-dim vector:
    # edge_binary[k, a] = 1 if aa a matches for edge k, else 0

    # We have multiple aa matches for the same (i,j), so we need to group.
    E = edge_index.shape[1]

    # Step 1: compress (i,j) pairs into unique edges
    edge_pairs = torch.stack([i, j], dim=1)  # (E, 2)
    unique_pairs, inverse = torch.unique(edge_pairs, dim=0, return_inverse=True)
    # unique edges: M, where M <= E

    M = unique_pairs.shape[0]  # number of unique edges

    # Step 2: create binary matrix
    edge_binary = torch.zeros((M, num_aa), dtype=torch.float32, device=device)
    edge_binary[inverse, aa_idx] = 1.0

    # Step 3: final edge_index for unique edges
    edge_index_final = unique_pairs.t().contiguous()  # (2, M)

    # Edge attributes are just the binary vectors
    edge_attr = edge_binary  # (M, 28)

    data = Data(
        x=pe_embeddings,
        edge_index=edge_index_final,
        edge_attr=edge_attr
    )
    return data

import torch
import torch.nn as nn
from torch_geometric.nn import GATv2Conv

class GNN_GATv2(nn.Module):
    """
    Graph encoder using GATv2Conv.

    Args:
        in_dim — input feature dimension (e.g., d_model = 64)
        hidden_dim — hidden layer dimension
        out_dim — final embedding dimension
        heads — number of attention heads
        dropout — attention dropout
    """

    def __init__(self, in_dim, hidden_dim=64, out_dim=64, heads=8, dropout=0.1):
        super().__init__()

        # First GATv2 layer
        self.gat1 = GATv2Conv(
            in_channels=in_dim,
            out_channels=hidden_dim // heads,
            heads=heads,
            dropout=dropout,
            concat=True
        )

        # Second GATv2 layer
        self.gat2 = GATv2Conv(
            in_channels=hidden_dim,
            out_channels=out_dim // heads,
            heads=heads,
            dropout=dropout,
            concat=True
        )

        self.act = nn.ReLU()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.gat1(x, edge_index) # (N, hidden_dim)
        x = self.act(x)

        x = self.gat2(x, edge_index) # (N, out_dim)

        return x


# GNN (used inside Transformer)
class GNN(nn.Module):
    def __init__(self, in_dim, hidden=64, out_dim=64):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden)
        self.conv2 = GCNConv(hidden, out_dim)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        # If no edges, returns transformed node features
        if edge_index is None or edge_index.numel() == 0:
            return x
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return x


class AAEmbedding(nn.Module):
    def __init__(self, device, embedding_dim=64, kernel_stride=7):
        super(AAEmbedding, self).__init__()
        self.embedding_dim = embedding_dim
        self.device = device
        self.amino_acids = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 
                            'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y',
                            'X', 'O','U', '1', '2', '3', '4', '5', '6', '7']

        self.aa_to_idx = {aa: idx for idx, aa in enumerate(self.amino_acids)}
        self.embedding_layer = nn.Embedding(num_embeddings=len(self.amino_acids), embedding_dim=self.embedding_dim).to(device)
        self.proteome_kernel = nn.Conv1d(in_channels=self.embedding_dim, out_channels=self.embedding_dim, kernel_size=kernel_stride*2+1, padding=kernel_stride, stride=kernel_stride).to(device)

    def forward(self, sequence):
        sequence_indices = torch.tensor([self.aa_to_idx[aa] for aa in sequence], device=self.device)
        embedded_sequence = self.embedding_layer(sequence_indices).unsqueeze(0)
        return embedded_sequence


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
        x = x.view(batch_size, -1, self.num_heads, self.depth)
        return x.permute(0, 2, 1, 3)

    def forward(self, q, k, v):
        batch_size = q.size(0)
        q = self.split_heads(self.wq(q), batch_size)
        k = self.split_heads(self.wk(k), batch_size)
        v = self.split_heads(self.wv(v), batch_size)
        attention_scores = F.softmax(torch.matmul(q, k.transpose(-2, -1)) / (self.depth ** 0.5), dim=-1)
        attention_output = torch.matmul(attention_scores, v)
        attention_output = attention_output.permute(0, 2, 1, 3).contiguous().view(batch_size, -1, self.d_model)
        return self.dense(attention_output)


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff=128):
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        return self.linear2(F.relu(self.linear1(x)))


class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads):
        super(DecoderLayer, self).__init__()
        self.conv1d = nn.Conv1d(in_channels=d_model, out_channels=d_model, kernel_size=5, padding=2)
        self.mha2 = MultiHeadAttention(d_model, num_heads)
        self.ffn = FeedForward(d_model)
        self.layernorm1 = nn.LayerNorm(d_model)
        self.layernorm2 = nn.LayerNorm(d_model)
        self.layernorm3 = nn.LayerNorm(d_model)

    def forward(self, x, enc_output):
        x = x.permute(0, 2, 1)
        conv_output = self.conv1d(x)
        conv_output = conv_output.permute(0, 2, 1)
        x = x.permute(0, 2, 1)
        x = self.layernorm1(x + conv_output)
        attn2 = self.mha2(x, enc_output, enc_output)
        x = self.layernorm2(x + attn2)
        ffn_output = self.ffn(x)
        return self.layernorm3(x + ffn_output)


class Decoder(nn.Module):
    def __init__(self, d_model, num_heads, num_layers):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList([DecoderLayer(d_model, num_heads) for _ in range(num_layers)])

    def forward(self, x, enc_output):
        for layer in self.layers:
            x = layer(x, enc_output)
        return x


class Transformer(nn.Module):
    def __init__(self, device, d_model=64, num_heads=8, num_layers=6, kernel_stride=7, graph_threshold=0.02):
        super(Transformer, self).__init__()
        self.device = device
        self.d_model = d_model
        self.graph_threshold = graph_threshold

        # replace positional encoder with graph-aware peak encoder
        self.encoder_positional_encoding = PeakEncodingWithDistances_Graph(d_model, device)

        # Transformer encoder (batch_first=True so tensors are (batch, seq_len, d_model))
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads, batch_first=True).to(device)
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers).to(device)

        # GNN to encode graph structure -> produces node embeddings of same dim
        #self.gnn = GNN(in_dim=d_model, hidden=d_model, out_dim=d_model).to(device)
        self.gnn = GNN_GATv2(
            in_dim=d_model,
            hidden_dim=128,
            out_dim=128,
            heads=8,
            dropout=0.1
        ).to(device)
        self.decoder_embedding = AAEmbedding(device, embedding_dim=d_model, kernel_stride=kernel_stride)
        self.decoder = Decoder(d_model, num_heads, num_layers).to(device)

        self.final_layer = nn.Linear(d_model, 1).to(device)

    def forward(self, encoder_input_mz, decoder_input_sequence):
        """
        encoder_input_mz: 1D tensor/list of mz peak positions (length N)
        decoder_input_sequence: string or iterable of amino-acid letters

        Returns: final_output shape (batch=1, seq_len_decoder, )
        """
        # Encode peaks -> pe (N, d_model) and match_matrices (adjacency) (28, N, N)
        pe, match_matrices = self.encoder_positional_encoding(encoder_input_mz)

        # Build graph 
        data = build_graph_fast(pe, match_matrices, threshold=self.graph_threshold)
        data = data.to(self.device)

        # GNN -> get node embeddings (N, d_model)
        node_embeddings = self.gnn(data) # (N, d_model)

        # transformer encoder on node embeddings (batch_first)
        enc_input = node_embeddings.unsqueeze(0)  # (1, N, d_model)
        enc_output = self.encoder(enc_input) # (1, N, d_model)

        # Decoder (AAEmbedding produces (1, seq_len_dec, d_model))
        decoder_input_embedded = self.decoder_embedding(decoder_input_sequence)  # (1, seq_len_dec, d_model)
        dec_output = self.decoder(decoder_input_embedded, enc_output)

        final_output = self.final_layer(dec_output).squeeze(-1) # (1, seq_len_dec)
        return final_output


class MultiTargetLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, prediction, targets):
        target_lse = torch.mean(prediction[targets], dim=0)
        lse = torch.logsumexp(prediction, dim=0)
        return lse - target_lse


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
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
    return logger


def generate_uuid():
    return base64.urlsafe_b64encode(uuid.uuid4().bytes).rstrip(b"=").decode("ascii")


def remove_brackets(input_string):
    return re.sub(r'\[.*?\]', '', input_string)


def add_random_letters(input_string):
    amino_acids = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 
                   'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
    random_letters_front = ''.join(random.choices(amino_acids, k=100))
    random_letters_back = ''.join(random.choices(amino_acids, k=100))
    return random_letters_front + input_string + random_letters_back
