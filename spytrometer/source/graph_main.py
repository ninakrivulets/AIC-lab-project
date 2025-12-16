import os, glob, pickle, sys, logging, argparse
import torch, torch.nn as nn
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from graph_bin_encoding import (
    Transformer,
    generate_uuid,
    get_logger,
    MultiTargetLoss,
)

GPU_ID = 0
BATCH_SIZE = 8
MAX_STEPS = int(10e7)
LEARNING_RATE = 1e-3

parser = argparse.ArgumentParser()
parser.add_argument('--resume_from', type=str, default=None, help='Resume training from experiment ID')
args, unknown = parser.parse_known_args()
resume_id = args.resume_from

device = torch.device(f"cuda:{GPU_ID}" if torch.cuda.is_available() else "cpu")

if resume_id:
    exp_id = resume_id
    exp_dir =  f"checkpoints/{exp_id}"
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

 
#logger.info("TensorBoard writer initialized.")



num_heads = 8
num_layers = 6
model_dim = 128
proteome_kernel_stride = 1 # Used in proteome embedding to reduce the proteom length.

model = Transformer(device=device,
                        num_heads=num_heads,
                        num_layers=num_layers,
                        d_model=model_dim, kernel_stride=proteome_kernel_stride).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20, eta_min=0.001)
loss_fn = nn.CrossEntropyLoss().to(device) # 
#loss_fn = MultiTargetLoss().to(device)

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
logger.info(f"Kernel stride in proteome embedding: {proteome_kernel_stride}")
if model_dim % num_heads:
    logger.error("Model_dim must be divisible by num_heads. Terminating with error")
    sys.exit(1)



with open("/home/data/Fasta/uniprot-proteome_UP000005640_canonical_isoforms.pkl", "rb") as f:
    prot_data = pickle.load(f)
protein_dict = prot_data["protein_dict"]      # {protein_id to AA‐string}
logger.info(f"Loaded {len(protein_dict)} proteins")

training_files = glob.glob("/blob/dda/PXD028806/training_data/*.pkl")
logger.info(f"Found {len(training_files)} training files")

while step < MAX_STEPS:
    logger.info(f"New epoch has started! Current step: {step}/{MAX_STEPS}")
    for fn in training_files:
        df = pickle.load(open(fn,"rb"))
        df = df.sample(frac=1).reset_index(drop=True)
        file_loss = 0.0
        logger.info(f"Reading training data file: {fn}, Data read: {len(df)}")
        for _, row in df.iterrows():

            peaks_mz = []
            peak_intensities = []
            peaks_mz.append(np.float64(1.007825035)) # mass of H
            peak_intensities.append(100) #intensity 100
            peaks_mz.extend(row.mz_values)
            intensities = row.intensity
            peak_intensities.extend(intensities)
            spectrum_neutral_mass = row.spectrum_neutral_mass
            spectral_min_mz = [(spectrum_neutral_mass - i) for i in row.mz_values]
            peaks_mz.extend(spectral_min_mz)
            #peak_intensities.extend([100]*150)
            peak_intensities.extend(intensities)
            peaks_mz.append(np.float64(1.007825035 + 15.99491463))   # mass of OH
            peaks_mz.append(np.float64(spectrum_neutral_mass))
            print("Before random", peaks_mz)
            max_peak = max(peaks_mz)
            min_peak = min(peaks_mz)
            size_peaks = len(peaks_mz)
            peaks_mz = np.random.uniform(low=min_peak, high=max_peak, size=size_peaks)
            #print("Before shuffling", peaks_mz[:15])
            #np.random.shuffle(peaks_mz)
            print("After random", peaks_mz)
            peak_intensities.append(100)

            peptide = row.sequence
            prot_ids = row.protein_ids  # {pid:{start,end}, ...}
            #spectrum_neutral_mass = row.spectrum_neutral_mass
            #spectral_min_mz = [(spectrum_neutral_mass - i) for i in row.mz_values]
            #peaks_mz.extend(spectral_min_mz)
            #peaks_mz.append(np.float64(1.007825035 + 15.99491463)) # mass of OH

            # print(len(peaks_mz)) is 302
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
            peptide_positions = list(range(int(pos["start"]//proteome_kernel_stride), int(pos["end"]//proteome_kernel_stride)))
            if len(peptide_positions) == 0:
              continue
            # logger.info(f"peptide_positions: {peptide_positions}, protein length {len(full_seq)}")
            # forward
            peaks_mz_tensor = torch.tensor(peaks_mz, dtype=torch.float32, device=device)

            # Forward pass through the model
            logits = model(peaks_mz_tensor, full_seq)
            #logits = model(peaks_mz, full_seq)  # to shape (len(full_seq),)
            logits = logits.view(1, -1)

            center = (pos['start'] + pos['end'])//2
            center_index = int(center //proteome_kernel_stride)
            target = torch.tensor([center_index], device=device)
            # target = torch.tensor(peptide_positions, device=device)
            #loss = loss_fn(logits, peptide_positions)
            loss = loss_fn(logits, target)
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


        if step >= MAX_STEPS:
            break
        logger.info(f"{os.path.basename(fn)}: mean loss = {file_loss/len(df):.4f}")
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
        }, os.path.join(exp_dir, 'model_checkpoint.pt'))

        with open(os.path.join(exp_dir, 'step.pkl'), 'wb') as f:
            pickle.dump(step, f)
    if scheduler is not None:
        scheduler.step()
    if step >= MAX_STEPS:
        break

logger.info("Training complete!")
writer.close()