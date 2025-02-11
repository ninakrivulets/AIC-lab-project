import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import pandas as pd
import os
import re
import pyopenms as oms
import matplotlib.pyplot as plt
import numpy as np
import random
import string
import uuid
import base64
import pickle
import sys
# from utils import get_logger, code_backup
from torch.utils.tensorboard import SummaryWriter
import shutil
from model import Transformer
from model import generate_uuid
from model import get_logger
from model import remove_brackets
from model import add_random_letters

if  __name__ == "__main__":

    exp_id = generate_uuid()
    exp_dir = '../checkpoints/'+exp_id

    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)

    plogger = get_logger(exp_id=exp_id, log_path=exp_dir)
    plogger.info(f"Experiment ID: {exp_id}")

    tensorboard_writer = SummaryWriter(log_dir=f"{exp_dir}")
    plogger.info("TensorBoard writer initialized.")


    GPU_ID = 0
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device("cuda:"+str(GPU_ID) if torch.cuda.is_available() else "cpu")
    plogger.info(f"Torch version: {torch.__version__}")
    plogger.info(f"Torch device: {device}")

    transformer_model = Transformer(device=device).to(device)
    optimizer = torch.optim.Adam(transformer_model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss().to(device)


    pickle_file_path = "/blob/dda/PXD028806/training_data/PXD028806_tailor_1.pkl"
    plogger.info(f"reading data from:  {pickle_file_path}")
            
    # Load data from the pickle file
    with open(pickle_file_path, "rb") as f:
        data = pickle.load(f)

    plogger.info(f"Data num read:  {len(data)}")

    training_step = 0
    for index, data_item in data.iterrows():

        peaks_mz = data_item.mz_values
        sequence = data_item.sequence
        sequence_pain = remove_brackets(data_item.sequence)

        plogger.info(f"Peptide seq: {sequence}")  
        plogger.info(f"Peptide sequence: {sequence_pain}")

        plogger.info(f"Speactrum peak num: {len(peaks_mz)}") 
        
        # Encoder input
        encoder_input = torch.tensor(peaks_mz, device=device, dtype=torch.float32)
        
        # Decoder input (sequence)
        decoder_input = add_random_letters(sequence_pain)        
        plogger.info(f"Decoder input sequence: {decoder_input}")

        # Forward pass
        
        output_distribution = transformer_model(encoder_input, decoder_input)
        plogger.info(f"Model output distribution: {output_distribution}")
       
        # Loss function
        
        target_position = len(decoder_input) // 2
        target = torch.tensor(target_position, device=device, dtype=torch.long).unsqueeze(dim=0)
        loss = loss_fn(output_distribution.view(-1, output_distribution.size(-1)), target)

        plogger.info(f"Calculated loss: {loss.item()}")
        loss_to_report = loss.detach().data.cpu().numpy()
        training_step += 1

        if np.isnan(loss_to_report):
            plogger.info(f"loss: {loss_to_report}")
            sys.exit(f"Loss function is nan - {loss_to_report}!")

        tensorboard_writer.add_scalar("Loss/train", loss_to_report, training_step)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    plogger.info("Completed backpropagation and optimization step.")
plogger.info("Done. Bye.")