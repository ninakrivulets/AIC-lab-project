import os, glob, pickle, sys, logging
import torch, torch.nn as nn
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from model_with_pos_encoding_n import (
    Transformer,
    generate_uuid,
    get_logger,
    MultiTargetLoss,
)

GPU_ID = 0
BATCH_SIZE = 8
MAX_STEPS = int(1e6)
LEARNING_RATE = 1e-3

device = torch.device(f"cuda:{GPU_ID}" if torch.cuda.is_available() else "cpu")
exp_id = generate_uuid()
exp_dir = f"home/ninak/checkpoints/{exp_id}"
os.makedirs(exp_dir, exist_ok=True)

logger = get_logger(exp_id=exp_id, log_path=exp_dir)
logger.info(f"Experiment ID: {exp_id}")
logger.info(f"Using device {device}")

writer = SummaryWriter(log_dir=exp_dir)
logger.info("TensorBoard writer initialized.")

num_heads = 4
num_layers = 3
model_dim = 32
if model_dim % num_heads:
    logger.error("Model_dim must be divisible by num_heads. Terminating with error")
    sys.exit(1)

model = Transformer(device=device,
                        num_heads=num_heads,
                        num_layers=num_layers,
                        d_model=model_dim).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
loss_fn = MultiTargetLoss().to(device)

logger.info(f"Model:\n{model}")
logger.info(f"Optimizer: {optimizer}")
logger.info(f"Loss: {loss_fn}")

with open("/home/ninak/data/protein_data.pkl", "rb") as f:
    prot_data = pickle.load(f)
protein_dict = prot_data["protein_dict"]      # {protein_id to AA‐string}
logger.info(f"Loaded {len(protein_dict)} proteins")

training_files = glob.glob("/blob/dda/PXD028806/training_data/*.pkl")
logger.info(f"Found {len(training_files)} training files")

step = 0
while step < MAX_STEPS:
    logger.info("New epoch has started!")
    for fn in training_files:
        df = pickle.load(open(fn,"rb"))
        df = df.sample(frac=1).reset_index(drop=True)
        file_loss = 0.0

        for _, row in df.iterrows():
            peaks_mz = row.mz_values
            peptide = row.sequence
            prot_ids = row.protein_ids  # {pid:{start,end}, ...}

            # train on the *first* mapping
            pid, pos = next(iter(prot_ids.items()))
            if pid not in protein_dict:
                continue

            full_seq = protein_dict[pid]
            #print('Sequence len', len(full_seq))
            pep_len  = len(peptide)
            # center index of peptide in protein if we want to predict center
            #center   = pos["start"] + pep_len//2
            peptide_start = pos["start"]
            # forward
            logits = model(peaks_mz, full_seq).squeeze(0)     # to shape (len(full_seq),)
            target = torch.tensor([peptide_start], device=device)

            loss = loss_fn(logits, target)
            file_loss += loss.item()

            loss.backward()
            step += 1
            if step % BATCH_SIZE == 0:
                optimizer.step()
                optimizer.zero_grad()
            if step >= MAX_STEPS:
                break
            
            writer.add_scalar("Loss/train", loss.item(), step)

        logger.info(f"{os.path.basename(fn)}: mean loss = {file_loss/len(df):.4f}")
        if step >= MAX_STEPS:
            break

    if step >= MAX_STEPS:
        break

logger.info("Training complete!")
writer.close()
