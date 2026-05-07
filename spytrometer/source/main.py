import os
import sys
import glob
import pickle
import argparse
import time
import numpy as np
import torch
import gc
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR

from model_spec_batch import (
    Transformer,
    generate_uuid,
    get_logger,
    compute_b_ions_modified,
    SpectrumEncoding,
)

GPU_ID              = 2
MINI_BATCH_SIZE     = 32          # number of samples per weight update
MAX_STEPS           = 2000
LEARNING_RATE       = 1e-4
ALPHA               = 0.1         # weight of auxiliary expected-abs-dist loss

parser = argparse.ArgumentParser()
parser.add_argument("--resume_from", type=str, default=None,
                    help="Resume training from experiment ID")
args, _ = parser.parse_known_args()
resume_id = args.resume_from

device = torch.device(f"cuda:{GPU_ID}" if torch.cuda.is_available() else "cpu")

if resume_id:
    exp_id  = resume_id
    exp_dir = f"./checkpoints/{exp_id}"
    os.makedirs(exp_dir, exist_ok=True)
    with open(os.path.join(exp_dir, "step.pkl"), "rb") as st:
        step = pickle.load(st)
else:
    exp_id  = generate_uuid()
    exp_dir = f"checkpoints/{exp_id}"
    os.makedirs(exp_dir, exist_ok=True)
    step = 0

logger = get_logger(exp_id=exp_id, log_path=exp_dir)
logger.info(f"Experiment ID : {exp_id}")
logger.info(f"Using device  : {device}")

writer = SummaryWriter(log_dir=exp_dir,
                       purge_step=step if resume_id else None)

num_heads              = 8
num_layers             = 4
model_dim              = 512
proteome_kernel_stride = 3

if model_dim % num_heads:
    logger.error("model_dim must be divisible by num_heads. Terminating.")
    sys.exit(1)

model = Transformer(
    device=device,
    num_heads=num_heads,
    num_layers=num_layers,
    d_model=model_dim,
    kernel_stride=proteome_kernel_stride,
).to(device)

optimizer = torch.optim.AdamW(
    model.parameters(), lr=LEARNING_RATE,
    betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-4,
)

num_epochs = 100
warmup    = LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=5)
cosine    = CosineAnnealingLR(optimizer, T_max=num_epochs - 5, eta_min=1e-6)
scheduler = SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[5])

loss_fn          = nn.CrossEntropyLoss().to(device)
spectrum_encoder = SpectrumEncoding(d_model=model_dim, device=device)

if resume_id:
    ckpt = torch.load(
        os.path.join(exp_dir, "model_checkpoint.pt"), map_location=device
    )
    model.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    logger.info("Loaded model and optimizer from checkpoint")
else:
    logger.info("Initialized model")

logger.info(f"Model:\n{model}")
logger.info(f"Optimizer: {optimizer}")
logger.info(f"Loss: {loss_fn}")
logger.info(
    f"num_heads={num_heads}  num_layers={num_layers}  "
    f"model_dim={model_dim}  mini_batch_size={MINI_BATCH_SIZE}  "
    f"kernel_stride={proteome_kernel_stride}"
)

with open(
    "/home/data/Fasta/uniprot-proteome_UP000005640_canonical_isoforms.pkl", "rb"
) as f:
    prot_data = pickle.load(f)
protein_dict = prot_data["protein_dict"]
logger.info(f"Loaded {len(protein_dict)} proteins")

training_files = glob.glob("/home/data/PDC000219/generated_data/*.pkl")
training_files = training_files[:20]
logger.info(f"Found {len(training_files)} training files")


def run_batch(
    batch_mz_peaks:      list,   # list of B 1-D float tensors
    batch_sequences:     list,   # list of B proteome strings
    batch_pep_positions: list,   # list of B lists-of-int  (target positions)
    batch_centers:       list,   # list of B ints          (center_strided)
) -> tuple:
    """
    Encodes all spectra in one SpectrumEncoding call, runs one forward pass,
    computes per-sample losses, sums them, and performs one backward + update.

    Returns:
        total_loss_val  : float  (sum of per-sample losses, for logging)
        accuracies      : list of float (|argmax - center| per sample)
    """
    #  1. Encode all spectra in one shot 
    #  padded_spectra : (B, max_pairs, d_model)
    #  spec_mask      : (B, max_pairs)  True = padding
    padded_spectra, spec_mask = spectrum_encoder(batch_mz_peaks)

    #  2. Forward pass through Transformer 
    #  logits_list : list of B tensors, each (1, prot_len_i)
    logits_list = model(padded_spectra, spec_mask, batch_sequences)

    #  3. Compute loss for every sample 
    total_loss = torch.tensor(0.0, device=device)
    accuracies = []

    for j, logits in enumerate(logits_list):
        logits_2d      = logits.view(1, -1)           # (1, prot_len)
        pep_positions  = batch_pep_positions[j]
        center_strided = batch_centers[j]

        # Cross-entropy averaged over all peptide positions
        ce_loss = torch.tensor(0.0, device=device)
        for tgt in pep_positions:
            ce_loss = ce_loss + loss_fn(
                logits_2d,
                torch.tensor([tgt], device=device),
            )
        ce_loss = ce_loss / len(pep_positions)

        # Auxiliary: expected absolute distance from the true center
        probs   = F.softmax(logits_2d, dim=-1)
        classes = torch.arange(
            logits_2d.size(-1), device=device, dtype=probs.dtype
        ).view(1, -1)
        exp_abs = (
            (probs * (classes - float(center_strided)).abs())
            .sum(-1).mean()
        )

        sample_loss = ce_loss + ALPHA * exp_abs
        total_loss  = total_loss + sample_loss

        with torch.no_grad():
            acc = (torch.argmax(logits_2d) - center_strided).abs().float()
            accuracies.append(acc.item())

    #---- 4. Single backward pass + weight update 
    mean_loss = total_loss / len(logits_list)
    mean_loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    optimizer.zero_grad()
    gc.collect()
    torch.cuda.empty_cache()
    return total_loss.item(), accuracies


train_start = time.time()
while step < MAX_STEPS:
    logger.info(f"New epoch started | step {step}/{MAX_STEPS}")
    skipped = 0

    for fn in training_files:
        #  Load file 
        try:
            if os.path.getsize(fn) == 0:
                logger.warning(f"Skipping empty file: {fn}")
                skipped += 1
                continue
            df = pickle.load(open(fn, "rb"))
        except (EOFError, pickle.UnpicklingError) as e:
            logger.warning(f"Skipping corrupted file {fn}: {e}")
            continue

        df = df.sample(frac=1).reset_index(drop=True)
        logger.info(f"Reading: {os.path.basename(fn)}  rows={len(df)}")

        file_loss    = 0.0
        accuracy_all = []

        # Mini-batch accumulators
        batch_mz_peaks:      list = []
        batch_sequences:     list = []
        batch_pep_positions: list = []
        batch_centers:       list = []

        for _, row in df.iterrows():
            prot_ids = row.protein_ids
            pid, pos = next(iter(prot_ids.items()))
            if pid not in protein_dict:
                continue

            full_seq = protein_dict[pid]

            peptide_positions = list(range(
                int(pos["start"] // proteome_kernel_stride),
                int(pos["end"]   // proteome_kernel_stride),
            ))
            if len(peptide_positions) == 0:
                continue

            center_strided = (
                (pos["start"] + pos["end"]) // (2 * proteome_kernel_stride)
            )
            peaks_mz = (
                [np.float64(1.007825035)]
                + compute_b_ions_modified(row.sequence)
            )
            peaks_t = torch.tensor(peaks_mz, dtype=torch.float32, device=device)

            #  Accumulate into batch 
            batch_mz_peaks.append(peaks_t)
            batch_sequences.append(full_seq)
            batch_pep_positions.append(peptide_positions)
            batch_centers.append(center_strided)

            #  When batch is full: encode + forward + backward 
            if len(batch_mz_peaks) == MINI_BATCH_SIZE:
                loss_val, accs = run_batch(
                    batch_mz_peaks,
                    batch_sequences,
                    batch_pep_positions,
                    batch_centers,
                )
                file_loss    += loss_val
                accuracy_all += accs
                step         += MINI_BATCH_SIZE

                writer.add_scalar("Loss/train",          loss_val / MINI_BATCH_SIZE, step)
                for acc in accs:
                    writer.add_scalar("Abs Distance/train", acc, step)

                # Clear accumulators
                batch_mz_peaks.clear()
                batch_sequences.clear()
                batch_pep_positions.clear()
                batch_centers.clear()

            if step >= MAX_STEPS:
                break

        #  Process leftover samples (partial last batch) 
        if batch_mz_peaks and step < MAX_STEPS:
            loss_val, accs = run_batch(
                batch_mz_peaks,
                batch_sequences,
                batch_pep_positions,
                batch_centers,
            )
            file_loss    += loss_val
            accuracy_all += accs
            step         += len(batch_mz_peaks)

            writer.add_scalar("Loss/train", loss_val / len(batch_mz_peaks), step)
            for acc in accs:
                writer.add_scalar("Abs Distance/train", acc, step)

            batch_mz_peaks.clear()
            batch_sequences.clear()
            batch_pep_positions.clear()
            batch_centers.clear()

        if step >= MAX_STEPS:
            break

        n_valid = max(len(df), 1)
        logger.info(
            f"{os.path.basename(fn)}: mean loss = {file_loss / n_valid:.4f}"
        )
        if accuracy_all:
            mean_acc = float(np.mean(accuracy_all))
            logger.info(
                f"{os.path.basename(fn)}: mean accuracy = {mean_acc:.4f}"
            )
        else:
            logger.warning(f"{os.path.basename(fn)}: no valid samples")
        logger.info(f"Skipped files so far: {skipped}")

        torch.save(
            {
                "model_state_dict":     model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            },
            os.path.join(exp_dir, "model_checkpoint.pt"),
        )
        with open(os.path.join(exp_dir, "step.pkl"), "wb") as fp:
            pickle.dump(step, fp)
        del df
        gc.collect()
        torch.cuda.empty_cache()

    # End of epoch
    scheduler.step()
    gc.collect()
    torch.cuda.empty_cache()
    if step >= MAX_STEPS:
        break

total_time    = time.time() - train_start
steps_per_sec = step / max(total_time, 1e-9)

logger.info("Training complete!")
logger.info(
    f"\n{'='*50}\n"
    f"  Total time   : {total_time:.1f}s  ({total_time/60:.1f} min)\n"
    f"  Total steps  : {step}\n"
    f"  Steps/second : {steps_per_sec:.2f}\n"
    f"{'='*50}"
)
