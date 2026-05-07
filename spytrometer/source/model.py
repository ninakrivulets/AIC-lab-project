import re
import math
import uuid
import logging
import torch
import time
import torch.nn as nn
import torch.nn.functional as F

AA_MASS = {
    "A": 71.03711,  "R": 156.10111, "N": 114.04293, "D": 115.02694,
    "C": 103.00919, "E": 129.04259, "Q": 128.05858, "G": 57.02146,
    "H": 137.05891, "I": 113.08406, "L": 113.08406, "K": 128.09496,
    "M": 131.04049, "F": 147.06841, "P": 97.05276,  "S": 87.03203,
    "T": 101.04768, "W": 186.07931, "Y": 163.06333, "V": 99.06841,
}

PROTON = 1.007276466812
Nterm  = 1.007825035


def generate_uuid():
    return uuid.uuid4().hex[:22].replace("-", "_")


def parse_modified_sequence(sequence: str):
    """Parse peptide sequence with inline PTMs like M[15.99]."""
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
    for mass in residue_masses:
        cumulative_mass += mass
        b_ions.append(cumulative_mass)
    return b_ions

class SpectrumEncoding(nn.Module):
    def __init__(self, d_model: int = 64, device: str = "cuda"):
        super().__init__()
        self.d_model = d_model
        self.device  = device

        AA_pair_mass = dict(AA_MASS)
        for k1, v1 in AA_MASS.items():
            for k2, v2 in AA_MASS.items():
                AA_pair_mass[k1 + k2] = v1 + v2

        self.register_buffer(
            "masses",
            torch.tensor(list(AA_pair_mass.values()), dtype=torch.float32, device=device),
        )
        self.AA_num = self.masses.shape[0]

        peak_pos_embedding_dim = d_model - self.AA_num
        if peak_pos_embedding_dim < 1:
            raise ValueError(f"d_model ({d_model}) must be > AA_num ({self.AA_num})")

        scale    = torch.tensor(0.001, dtype=torch.float32, device=device)
        base     = torch.tensor(5000.0, dtype=torch.float32, device=device)
        exp_term = (
            torch.arange(1, peak_pos_embedding_dim + 1, 2, dtype=torch.float32, device=device)
            * 2.0 / peak_pos_embedding_dim
        )
        self.register_buffer("div_term", 1.0 / (scale * torch.pow(base, exp_term)))
        self._triu_cache: dict = {}

    def _get_triu(self, P: int) -> torch.Tensor:
        if P not in self._triu_cache:
            self._triu_cache[P] = torch.triu(
                torch.ones(P, P, dtype=torch.bool, device=self.masses.device),
                diagonal=1,
            )
        return self._triu_cache[P]

    def forward(self, mz_peaks_list: list) -> tuple:
        dev = self.masses.device
        B            = len(mz_peaks_list)
        peak_lengths = [t.shape[0] for t in mz_peaks_list]
        max_peaks    = max(peak_lengths)
        P            = max_peaks

        # Step 1 – pad raw m/z peaks to (B, P)
        peaks_padded = torch.zeros(B, P, dtype=torch.float32, device=dev)
        peak_valid   = torch.zeros(B, P, dtype=torch.bool,    device=dev)
        for b, t in enumerate(mz_peaks_list):
            L = t.shape[0]
            peaks_padded[b, :L] = t.to(dev)
            peak_valid[b, :L]   = True

        # Step 2 – sinusoidal encoding  (B, P, d_model - AA_num)
        outer = peaks_padded.unsqueeze(-1) * self.div_term   # (B, P, half_dim)
        pe    = torch.cat([torch.sin(outer), torch.cos(outer)], dim=-1)
        # pe : (B, P, d_model - AA_num)

        # Step 3 – pairwise distances  (B, P, P)
        diff = (peaks_padded.unsqueeze(2) - peaks_padded.unsqueeze(1)).abs()

        # Step 4 – AA fingerprint  (B, P, P, AA_num)
        diff_exp = diff.unsqueeze(-1)
        mass_exp = self.masses.view(1, 1, 1, -1)
        aa_match = (diff_exp - mass_exp).abs() < 0.02   # (B, P, P, AA) bool

        # Step 5 – valid pair mask  (B, P, P)
        triu_mask = self._get_triu(P)
        both_real = peak_valid.unsqueeze(2) & peak_valid.unsqueeze(1)
        any_match = aa_match.any(dim=-1)
        pair_mask = triu_mask.unsqueeze(0) & both_real & any_match  # (B, P, P)

        pair_mask_flat = pair_mask.view(B, P * P)          # (B, P²)

        # All valid (batch, flat_pair) indices — one GPU op, no loop
        valid_b, valid_flat = pair_mask_flat.nonzero(as_tuple=True)  # (N_valid,)

        valid_i = valid_flat // P    # row index  (N_valid,)
        valid_j = valid_flat  % P    # col index  (N_valid,)

        pe_selected = pe[valid_b, valid_i]                 # (N_valid, d-AA)

        aa_selected = aa_match[valid_b, valid_i, valid_j].float()  # (N_valid, AA)
        all_valid_emb = torch.cat([pe_selected, aa_selected], dim=-1)

        # Step 7 – pad to (B, max_valid, d_model) using cumsum trick
        #          to avoid any Python loop over samples
        # Count valid pairs per sample
        counts   = pair_mask_flat.sum(dim=1)               # (B,)  int
        max_valid = int(counts.max().item())

        if max_valid == 0:
            out_padded = torch.zeros(B, 1, self.d_model, device=dev)
            out_mask   = torch.ones(B, 1, dtype=torch.bool, device=dev)
            return out_padded, out_mask

        sample_offsets     = torch.zeros(B, dtype=torch.long, device=dev)
        sample_offsets[1:] = counts[:-1].cumsum(dim=0)        # (B,)
        within_pos         = (
            torch.arange(valid_b.shape[0], device=dev)
            - sample_offsets[valid_b]
        )                                                       # (N_valid,)

        out_padded = torch.zeros(B, max_valid, self.d_model, device=dev)
        out_padded[valid_b, within_pos] = all_valid_emb

        positions = torch.arange(max_valid, device=dev).unsqueeze(0)  # (1, max_valid)
        out_mask  = positions >= counts.unsqueeze(1)                   # (B, max_valid)

        return out_padded, out_mask   # (B, max_valid, d_model), (B, max_valid)


'''
# ---------------------------------------------------------------------------
# Spectrum Encoding  –  processes the WHOLE BATCH at once
# ---------------------------------------------------------------------------
class SpectrumEncoding(nn.Module):
    """
    Encodes a batch of spectra in one forward pass.

    forward(mz_peaks_list) where mz_peaks_list is a list of B 1-D tensors,
    each containing the m/z values for one spectrum.

    Returns:
        padded_batch : (B, max_pairs, d_model)   – zero-padded
        pad_mask     : (B, max_pairs)             – True = padding position
    """

    def __init__(self, d_model: int = 64, device: str = "cuda"):
        super().__init__()
        self.d_model = d_model
        self.device  = device

        # Build single-AA + pair-AA mass table
        AA_pair_mass = dict(AA_MASS)
        for k1, v1 in AA_MASS.items():
            for k2, v2 in AA_MASS.items():
                AA_pair_mass[k1 + k2] = v1 + v2

        self.masses = torch.tensor(
            list(AA_pair_mass.values()), dtype=torch.float32, device=device
        )
        self.AA_num = self.masses.shape[0]

        scale = torch.tensor(0.001, dtype=torch.float32, device=device)
        base  = torch.tensor(5000.0, dtype=torch.float32, device=device)

        peak_pos_embedding_dim = d_model - self.AA_num
        if peak_pos_embedding_dim < 1:
            raise ValueError(
                f"d_model ({d_model}) must be > AA_num ({self.AA_num})"
            )

        exp_term = (
            torch.arange(1, peak_pos_embedding_dim + 1, 2,
                         dtype=torch.float32, device=device)
            * 2.0 / peak_pos_embedding_dim
        )
        # shape: (peak_pos_embedding_dim // 2,)
        self.div_term = 1.0 / (scale * torch.pow(base, exp_term))

    # ------------------------------------------------------------------
    def _encode_single(self, mz_peaks: torch.Tensor) -> torch.Tensor:
        """
        Encode one spectrum.
        Args:
            mz_peaks : 1-D float tensor, shape (n_peaks,)
        Returns:
            2-D tensor, shape (n_valid_pairs, d_model)
            If no valid pairs exist, returns zeros of shape (1, d_model).
        """
        mz_peaks = mz_peaks.to(self.device)

        # ---- Sinusoidal peak-position encoding -------------------------
        outer  = torch.outer(mz_peaks, self.div_term)     # (n, half_dim)
        pe_sin = torch.sin(outer)
        pe_cos = torch.cos(outer)
        # (n_peaks, d_model - AA_num)
        spectrum_tensor = torch.cat((pe_sin, pe_cos), dim=1)

        # ---- AA-distance binary fingerprint ----------------------------
        diff_AA_idx = self._peak_distances(mz_peaks)      # (n, n, AA_num)
        diff_AA_cnt = diff_AA_idx.sum(dim=2)              # (n, n)
        peak_pair_idx = diff_AA_cnt.nonzero()             # (K, 2)

        rows = []
        for pair in peak_pair_idx:
            if pair[0] < pair[1]:
                rows.append(
                    torch.cat((
                        spectrum_tensor[pair[0]],          # (d_model-AA_num,)
                        diff_AA_idx[pair[0], pair[1]],     # (AA_num,)
                    ))                                     # → (d_model,)
                )

        if len(rows) == 0:
            return torch.zeros(1, self.d_model, device=self.device)

        return torch.stack(rows)   # (n_valid_pairs, d_model)

    # ------------------------------------------------------------------
    def forward(
        self,
        mz_peaks_list: list,
    ) -> tuple:
        """
        Encode a whole batch of spectra at once.

        Args:
            mz_peaks_list : list of B 1-D float tensors,
                            mz_peaks_list[i] has shape (n_peaks_i,)

        Returns:
            padded_batch : (B, max_pairs, d_model)   float, zero-padded
            pad_mask     : (B, max_pairs)             bool, True = pad token
        """
        # Step 1 – encode every spectrum independently (vectorised within
        #           each spectrum; different spectra have different lengths
        #           so we cannot fuse them before the pair-selection step)
        encoded = [self._encode_single(mz) for mz in mz_peaks_list]
        # encoded[i] : (n_pairs_i, d_model)

        B       = len(encoded)
        lengths = [e.shape[0] for e in encoded]
        max_len = max(lengths)

        # Step 2 – pad to (B, max_pairs, d_model) and build mask
        padded = torch.zeros(B, max_len, self.d_model, device=self.device)
        mask   = torch.ones(B, max_len, dtype=torch.bool, device=self.device)
        # mask: True = padding (convention used by PyTorch MHA / TransformerEncoder)

        for i, enc in enumerate(encoded):
            L = lengths[i]
            padded[i, :L, :] = enc
            mask[i, :L]      = False   # False = real token

        return padded, mask   # (B, max_pairs, d_model), (B, max_pairs)

    # ------------------------------------------------------------------
    def _peak_distances(self, mz_peaks: torch.Tensor) -> torch.Tensor:
        """
        Returns diff_AA_idx[i, j, k] = 1 if |peak_i - peak_j| ≈ mass_k.
        Shape: (n_peaks, n_peaks, AA_num)
        """
        n         = mz_peaks.shape[0]
        peaks_vec = mz_peaks.unsqueeze(1)                          # (n, 1)
        diff      = torch.cdist(peaks_vec, peaks_vec)              # (n, n)

        diff_exp  = diff.unsqueeze(2).expand(-1, -1, self.AA_num)  # (n,n,AA)
        mass_exp  = (
            self.masses
            .unsqueeze(0).expand(n, -1)
            .unsqueeze(0).expand(n, -1, -1)
        )                                                          # (n,n,AA)

        diff_AA_idx = torch.where(
            torch.abs(mass_exp - diff_exp) < 0.02,
            torch.ones_like(diff_exp),
            torch.zeros_like(diff_exp),
        )
        return diff_AA_idx
'''
class AAEmbedding(nn.Module):
    def __init__(self, device, embedding_dim: int = 64, kernel_stride: int = 7):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.device        = device

        self.amino_acids = [
            "A", "C", "D", "E", "F", "G", "H", "I", "K", "L",
            "M", "N", "P", "Q", "R", "S", "T", "V", "W", "Y",
            "X", "O", "U", "1", "2", "3", "4", "5", "6", "7",
        ]
        self.aa_to_idx = {aa: idx for idx, aa in enumerate(self.amino_acids)}

        self.embedding_layer = nn.Embedding(
            num_embeddings=len(self.amino_acids),
            embedding_dim=self.embedding_dim,
        ).to(device)

        self.proteome_kernel = nn.Conv1d(
            in_channels=self.embedding_dim,
            out_channels=self.embedding_dim,
            kernel_size=kernel_stride * 2 + 1,
            padding=kernel_stride,
            stride=kernel_stride,
        ).to(device)

    def forward(self, sequence: str) -> torch.Tensor:
        """
        Returns (1, seq_len, embedding_dim).
        """
        indices = torch.tensor(
            [self.aa_to_idx[aa] for aa in sequence], device=self.device
        )
        embedded = self.embedding_layer(indices).unsqueeze(0)  # (1, L, d)
        # Apply stride Conv BEFORE returning — reduces L by kernel_stride
        embedded = embedded.permute(0, 2, 1)                   # (1, d, L)
        embedded = self.proteome_kernel(embedded)              # (1, d, L//stride)
        embedded = embedded.permute(0, 2, 1)                   # (1, L//stride, d)
        return embedded
        #return self.embedding_layer(indices).unsqueeze(0)   # (1, L, d)

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.d_model   = d_model
        self.depth     = d_model // num_heads

        self.wq    = nn.Linear(d_model, d_model)
        self.wk    = nn.Linear(d_model, d_model)
        self.wv    = nn.Linear(d_model, d_model)
        self.dense = nn.Linear(d_model, d_model)

    def _split_heads(self, x: torch.Tensor, B: int) -> torch.Tensor:
        return x.view(B, -1, self.num_heads, self.depth).permute(0, 2, 1, 3)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B = q.size(0)
        q = self._split_heads(self.wq(q), B)
        k = self._split_heads(self.wk(k), B)
        v = self._split_heads(self.wv(v), B)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.depth)
        if key_padding_mask is not None:
            # key_padding_mask: (B, Sk) → True means ignore
            scores = scores.masked_fill(
                key_padding_mask[:, None, None, :], float("-inf")
            )

        attn   = F.softmax(scores, dim=-1)
        output = torch.matmul(attn, v)
        output = (
            output.permute(0, 2, 1, 3)
            .contiguous()
            .view(B, -1, self.d_model)
        )
        return self.dense(output)


class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int = 2048):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(F.relu(self.linear1(x)))


class DecoderLayer(nn.Module):
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        self.conv1d     = nn.Conv1d(d_model, d_model, kernel_size=5, padding=2)
        self.mha2       = MultiHeadAttention(d_model, num_heads)
        self.ffn        = FeedForward(d_model)
        self.layernorm1 = nn.LayerNorm(d_model)
        self.layernorm2 = nn.LayerNorm(d_model)
        self.layernorm3 = nn.LayerNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        enc_output: torch.Tensor,
        enc_key_padding_mask: torch.Tensor | None = None,
        tgt_key_padding_mask=None
    ) -> torch.Tensor:
        # Conv along sequence dimension
        conv_out = self.conv1d(x.permute(0, 2, 1)).permute(0, 2, 1)
        x        = self.layernorm1(x + conv_out)

        # Cross-attention: query = proteome, key/value = spectrum encoder
        attn2 = self.mha2(x, enc_output, enc_output,
                          key_padding_mask=enc_key_padding_mask)
        x     = self.layernorm2(x + attn2)

        return self.layernorm3(x + self.ffn(x))


class Decoder(nn.Module):
    def __init__(self, d_model: int, num_heads: int, num_layers: int):
        super().__init__()
        self.layers = nn.ModuleList(
            [DecoderLayer(d_model, num_heads) for _ in range(num_layers)]
        )

    def forward(
        self,
        x: torch.Tensor,
        enc_output: torch.Tensor,
        enc_key_padding_mask: torch.Tensor | None = None,
        tgt_key_padding_mask = None
    ) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, enc_output,
                      enc_key_padding_mask=enc_key_padding_mask,
                      tgt_key_padding_mask=tgt_key_padding_mask)
            #x = layer(x, enc_output, enc_key_padding_mask)
        return x

lass Transformer(nn.Module):
    """
    Batched transformer.

    forward(
        spectrum_embeddings,   # (B, max_spec_len, d_model)  – from SpectrumEncoding
        spec_padding_mask,     # (B, max_spec_len)            – True = pad
        sequences,             # list[str]  length B          – proteome strings
    ) -> list of B tensors, each (1, prot_len_i_strided)
    """

    def __init__(
        self,
        device,
        d_model:       int = 64,
        num_heads:     int = 8,
        num_layers:    int = 6,
        kernel_stride: int = 3,
    ):
        super().__init__()
        self.d_model = d_model
        self.device  = device

        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=num_heads, batch_first=True
        ).to(device)
        self.encoder = nn.TransformerEncoder(
            self.encoder_layer, num_layers=num_layers
        ).to(device)

        self.decoder_embedding = AAEmbedding(
            device, embedding_dim=d_model, kernel_stride=kernel_stride
        )
        self.decoder = Decoder(d_model, num_heads, num_layers).to(device)

        self.final_layer = nn.Conv1d(
            in_channels=d_model, out_channels=1, kernel_size=7, padding=3
        ).to(device)

    def forward(
        self,
        spectrum_embeddings: torch.Tensor,
        spec_padding_mask:   torch.Tensor,
        sequences:           list,
    ) -> list:
        """
        Returns a list of B tensors, each of shape (1, prot_len_i).
        Proteomes differ in length so the decoder runs per-sample,
        while the encoder runs over the whole batch in one shot.
        """
        B = spectrum_embeddings.size(0)
        #  Encode all spectra together 
        # (B, max_spec_len, d_model)  ->  (B, max_spec_len, d_model)
        enc_output = self.encoder(
            spectrum_embeddings,
            src_key_padding_mask=spec_padding_mask,
        )
        # NEW
        #  Embed ALL proteomes and pad to same length 
        embedded_seqs = [self.decoder_embedding(s) for s in sequences]
        # embedded_seqs[i]: (1, L_i // stride, d_model)  after strided conv

        seq_lengths = [e.shape[1] for e in embedded_seqs]
        max_seq_len = max(seq_lengths)

        # Pad to (B, max_seq_len, d_model)
        prot_padded = torch.zeros(
            B, max_seq_len, self.d_model, device=self.device
        )
        prot_mask = torch.ones(
            B, max_seq_len, dtype=torch.bool, device=self.device
        )
        for i, emb in enumerate(embedded_seqs):
            L = seq_lengths[i]
            prot_padded[i, :L] = emb.squeeze(0)
            prot_mask[i, :L]   = False
        
        #  Run decoder over the WHOLE batch in one shot 
        dec_out = self.decoder(
            prot_padded,    # (B, max_seq_len, d_model)
            enc_output,     # (B, max_spec_len, d_model)
            enc_key_padding_mask=spec_padding_mask,
            tgt_key_padding_mask=prot_mask,        # mask padding in proteome too
        )  # (B, max_seq_len, d_model)

        #  Final projection — batched Conv1d 
        dec_out_t    = dec_out.permute(0, 2, 1)         # (B, d_model, max_seq_len)
        logits_padded = self.final_layer(dec_out_t)      # (B, 1, max_seq_len)
        logits_padded = logits_padded.squeeze(1)         # (B, max_seq_len)

        # Return per-sample logits trimmed to actual length
        outputs = [
            logits_padded[i, :seq_lengths[i]].unsqueeze(0)
            for i in range(B)
        ]
        '''
        #  Decode each sample (variable-length proteomes) 
        outputs = []
        for i, prot_seq in enumerate(sequences):
            prot_emb = self.decoder_embedding(prot_seq)        # (1, L, d)
            enc_i    = enc_output[i].unsqueeze(0)              # (1, max_spec, d)
            mask_i   = spec_padding_mask[i].unsqueeze(0)       # (1, max_spec)

            dec_out  = self.decoder(prot_emb, enc_i, mask_i)   # (1, L, d)

            logit_i  = (
                self.final_layer(dec_out.permute(0, 2, 1))     # (1, 1, L)
                .squeeze(1)                                     # (1, L)
            )
            outputs.append(logit_i)
        '''
        return outputs   # list of B tensors, each (1, prot_len_i)

def get_logger(exp_id: str, log_path: str):
    logger = logging.getLogger("main")
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        fh  = logging.FileHandler(f"{log_path}/experiment_{exp_id}.log")
        ch  = logging.StreamHandler()
        fmt = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s "
            "[%(filename)s:%(lineno)d %(funcName)s]"
        )
        for h in (fh, ch):
            h.setLevel(logging.INFO)
            h.setFormatter(fmt)
            logger.addHandler(h)
    return logger

def remove_brackets(s):
    return re.sub(r"\[.*?\]", "", s)


def add_random_letters(s):
    aas = list(AA_MASS.keys())
    return (
        "".join(random.choices(aas, k=100))
        + s
        + "".join(random.choices(aas, k=100))
    )

