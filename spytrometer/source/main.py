import os, glob, pickle, sys, logging
import torch, torch.nn as nn
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from model_with_pos_encoding import (
    Transformer,
    generate_uuid,
    get_logger,
    MultiTargetLoss,
)

GPU_ID = 0
BATCH_SIZE = 8
MAX_STEPS = int(2e6)
LEARNING_RATE = 2e-3

device = torch.device(f"cuda:{GPU_ID}" if torch.cuda.is_available() else "cpu")
exp_id = generate_uuid()
exp_dir = f"./../checkpoints/{exp_id}"
os.makedirs(exp_dir, exist_ok=True)

logger = get_logger(exp_id=exp_id, log_path=exp_dir)
logger.info(f"Experiment ID: {exp_id}")
logger.info(f"Using device {device}")

writer = SummaryWriter(log_dir=exp_dir)
logger.info("TensorBoard writer initialized.")

num_heads = 8
num_layers = 6
model_dim = 128
poteome_kernel_stride = 1 # Used in proteome embedding to reduce the proteom length.
logger.info(f"Model num head: {num_heads}")
logger.info(f"Model num layer: {num_layers}")
logger.info(f"Model dim : {model_dim}")
logger.info(f"Kernel stride in proteome embedding: {poteome_kernel_stride}")
if model_dim % num_heads:
    logger.error("Model_dim must be divisible by num_heads. Terminating with error")
    sys.exit(1)

model = Transformer(device=device,
                        num_heads=num_heads,
                        num_layers=num_layers,
                        d_model=model_dim, kernel_stride=poteome_kernel_stride).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
# loss_fn = nn.CrossEntropyLoss().to(device) # 
loss_fn = MultiTargetLoss().to(device)

logger.info(f"Model:\n{model}")
logger.info(f"Optimizer: {optimizer}")
logger.info(f"Loss: {loss_fn}")

with open("/home/data/Fasta/uniprot-proteome_UP000005640_canonical_isoforms.pkl", "rb") as f:
    prot_data = pickle.load(f)
protein_dict = prot_data["protein_dict"]      # {protein_id to AA‐string}
logger.info(f"Loaded {len(protein_dict)} proteins")

training_files = glob.glob("/blob/dda/PXD028806/training_data/*.pkl")
logger.info(f"Found {len(training_files)} training files")

step = 0
while step < MAX_STEPS:
    logger.info(f"New epoch has started! Current steps: {step} out of max step: {MAX_STEPS}")
    for fn in training_files:
        df = pickle.load(open(fn,"rb"))
        df = df.sample(frac=1).reset_index(drop=True)
        file_loss = 0.0
        logger.info(f"Reading training data file:  {fn}, Data read: {len(df)}")
        for _, row in df.iterrows():
            peaks_mz = row.mz_values
            peptide = row.sequence
            prot_ids = row.protein_ids  # {pid:{start,end}, ...}

            # print(peaks_mz)
            # print(peptide)
            # print(prot_ids)

            # train on the *first* mapping
            pid, pos = next(iter(prot_ids.items()))
            if pid not in protein_dict:
                continue

            # print(pos["start"])
            # print(protein_dict[pid])
            full_seq = protein_dict[pid]+"ACDKLNACDKLNACDKLNACDKLNACDKLNACDKLN"
            # print('Sequence len', len(full_seq))
            # logger.info(f"protein:  {full_seq}")
            # pep_len  = len(peptide)
            # center index of peptide in protein if we want to predict center
            #center   = pos["start"] + pep_len//2
            # peptide_start = pos["start"]//poteome_kernel_stride
            peptide_positions = list(range(int(pos["start"]//poteome_kernel_stride), int(pos["end"]//poteome_kernel_stride)))
            if len(peptide_positions) == 0:
              continue
            # logger.info(f"peptide_positions: {peptide_positions}, protein length {len(full_seq)}")
            # forward
            logits = model(peaks_mz, full_seq).squeeze(0)     # to shape (len(full_seq),)
            # target = torch.tensor(peptide_positions, device=device)
            loss = loss_fn(logits, peptide_positions)

            # target = torch.tensor((pos["start"]+pos["end"])//2, device=device, dtype=torch.long).unsqueeze(dim=0)
            # loss = loss_fn(logits.unsqueeze(0), target)


            file_loss += loss.item()
            # logger.info(f"Model output distribution: {logits}, loss: {loss.item()}")
            # logger.info(f"Model output distribution: , loss: {loss.item()}")

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
