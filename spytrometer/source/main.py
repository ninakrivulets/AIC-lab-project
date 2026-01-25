import os, glob, pickle, sys, logging, argparse
import torch, torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.tensorboard import SummaryWriter

from model_with_pos_encoding import (
    Transformer,
    generate_uuid,
    get_logger,
    compute_b_ions_modified,
    SpectrumEncoding,
)

GPU_ID = 0
BATCH_SIZE = 32
MAX_STEPS = int(10e7)
LEARNING_RATE = 1e-4

parser = argparse.ArgumentParser()
parser.add_argument('--resume_from', type=str, default=None, help='Resume training from experiment ID')
args, unknown = parser.parse_known_args()
resume_id = args.resume_from

device = torch.device(f"cuda:{GPU_ID}" if torch.cuda.is_available() else "cpu")

if resume_id:
    exp_id = resume_id
    exp_dir =  f"./checkpoints/{exp_id}"
    os.makedirs(exp_dir, exist_ok=True)
    step_file = os.path.join(exp_dir, 'step.pkl')
    with open(step_file, 'rb') as st:
        step = pickle.load(st)
else:
    exp_id = generate_uuid()
    exp_dir = f"checkpoints/{exp_id}"
#exp_dir = f"./../checkpoints/{exp_id}"
    os.makedirs(exp_dir, exist_ok=True)
    step = 0

logger = get_logger(exp_id=exp_id, log_path=exp_dir)
logger.info(f"Experiment ID: {exp_id}")
logger.info(f"Using device {device}")

writer = SummaryWriter(log_dir=exp_dir)

num_heads = 8
num_layers = 4
model_dim = 40
proteome_kernel_stride = 1 # Used in proteome embedding to reduce the proteom length.

model = Transformer(device=device,
                        num_heads=num_heads,
                        num_layers=num_layers,
                        d_model=model_dim, kernel_stride=proteome_kernel_stride).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-4)
loss_fn = nn.CrossEntropyLoss().to(device) # 
# loss_fn = MultiTargetLoss().to(device)
spectrum_encoder = SpectrumEncoding(d_model=model_dim)

if resume_id:
    checkpoint_path = os.path.join(exp_dir, 'model_checkpoint.pt')
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    logger.info('Loaded model and optimizer from checkpoint')
else:
    logger.info('Initialized model')

logger.info(f"Model:\n{model}")
logger.info(f"Optimizer: {optimizer}")
logger.info(f"Loss: {loss_fn}")

logger.info(f"Model num head: {num_heads}")
logger.info(f"Model num layer: {num_layers}")
logger.info(f"Model dim : {model_dim}")
logger.info(f"Mini batch size : {BATCH_SIZE}")
logger.info(f"Kernel stride in proteome embedding: {proteome_kernel_stride}")
if model_dim % num_heads:
    logger.error("Model_dim must be divisible by num_heads. Terminating with error")
    sys.exit(1)



with open("/home/data/Fasta/uniprot-proteome_UP000005640_canonical_isoforms.pkl", "rb") as f:
    prot_data = pickle.load(f)
protein_dict = prot_data["protein_dict"]      # {protein_id to AA‐string}
logger.info(f"Loaded {len(protein_dict)} proteins")

training_files = glob.glob("/blob/dda/PXD028806/training_data/*.pkl")
#training_files_path = '/home/ninak/checkpoints/_kU4hfpfT_iWbo9ZIW3m1A/PXD028806_tailor_177_0_RPEIVVATPGR.pkl'
#with open(training_files_path, 'rb') as file:
    # Load the data (deserialize)
    #training_files = pickle.load(file)
logger.info(f"Found {len(training_files)} training files")

while step < MAX_STEPS:
    logger.info(f"New epoch has started! Current step: {step}/{MAX_STEPS}")
    for fn in training_files:
        df = pickle.load(open(fn,"rb"))
        df = df.sample(frac=1).reset_index(drop=True)
        file_loss = 0.0
        logger.info(f"Reading training data file: {fn}, Data read: {len(df)}")
        for _, row in df.iterrows():
            peptide = row.sequence
            # print(peptide)
            peaks_mz = []
            peak_intensities = []
            peaks_mz.append(np.float64(1.007825035))  # start from proton N-term
            peaks_mz.extend(compute_b_ions_modified(peptide))

            spect_embed = spectrum_encoder(torch.tensor(peaks_mz, dtype=torch.float32, device=device))
            # exit()

            prot_ids = row.protein_ids  # {pid:{start,end}, ...}
            pid, pos = next(iter(prot_ids.items()))
            if pid not in protein_dict:
                continue

            full_seq = protein_dict[pid]+"AADFGHRELN"
            peptide_positions = list(range(int(pos["start"]//proteome_kernel_stride), int(pos["end"]//proteome_kernel_stride)))

            if len(peptide_positions) == 0:
              continue
            logits = model(spect_embed, full_seq)
            logits = logits.view(1, -1)
            center = (pos['start'] + pos['end'])//2
            # print(center)

            # print(logits)
            # exit()

            # center = (pos['start'] + pos['end'])//2
            # center_index = int(center //proteome_kernel_stride)
            # target = torch.tensor(peptide_positions, device=device)
            # if center_index >= logits.shape[1]:
            #     print("ERROR:", center_index, logits.shape[1], full_seq, len(full_seq))
            #     print("start/end:", pos['start'], pos['end'])
            #     print("stride:", proteome_kernel_stride)
            #     raise ValueError("Target index out of range")

            loss = torch.tensor([0.0], device=device)
            for pos in peptide_positions:
                loss += loss_fn(logits, torch.tensor([pos], device=device))
            loss = loss/torch.tensor([len(peptide_positions)*1.0], device=device)

            # print(loss.item())
            # loss = loss_fn(logits.unsqueeze(0), target)

            
            file_loss += loss.item()

            loss.backward()
            # exit()
            step += 1
            if step % BATCH_SIZE == 0:
                optimizer.step()
                optimizer.zero_grad()
            if step >= MAX_STEPS:
                break
            
            writer.add_scalar("Loss/train", loss.item(), step)


        if step >= MAX_STEPS:
            break
        logger.info(f"{os.path.basename(fn)}: mean loss = {file_loss/len(df):.4f}")
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
        }, os.path.join(exp_dir, 'model_checkpoint.pt'))

        with open(os.path.join(exp_dir, 'step.pkl'), 'wb') as f:
            pickle.dump(step, f)

    if step >= MAX_STEPS:
        break

logger.info("Training complete!")
writer.close()