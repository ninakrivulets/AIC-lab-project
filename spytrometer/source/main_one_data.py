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
from utils import get_logger, code_backup
from torch.utils.tensorboard import SummaryWriter
import shutil
from model import Transformer


def get_logger(exp_id, log_path):
    #logger = logging.getLogger(f"Experiment_{exp_id}")
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

        #formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

    return logger

def generate_uuid():
    """Generates a uuid 4 string, in this context for tracking each run of the experiment
    Returns:
        an ascii friendly uuid4 string.
    """
    return base64.urlsafe_b64encode(uuid.uuid4().bytes).rstrip(b"=").decode("ascii")

def remove_brackets(input_string):
    return re.sub(r'\[.*?\]', '', input_string)

def add_random_letters(input_string):
    amino_acids = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 
                            'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
    # Generate 100 random capital letters
    random_letters_front = ''.join(random.choices(amino_acids, k=100))
    random_letters_back = ''.join(random.choices(amino_acids, k=100))
    # Add random letters before and after the input string
    return random_letters_front + input_string + random_letters_back



if  __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    transformer_model = Transformer(device=device).to(device)
    '''
    for name, param in transformer_model.named_parameters():
        print(f"Parameter name: {name}")
        if param is not None:
            print(param)
    '''
        #if not param.requires_grad:
            #print(f"Parameter {param} has requires_grad=False. Setting it to True.")
            #param.requires_grad = True

    #optimizer = torch.optim.Adam(transformer_model.parameters(), lr=1e-3)
    optimizer = torch.optim.Adam(transformer_model.parameters(), lr=0.1)

    pickle_file_path = "/blob/dda/PXD028806/training_data/PXD028806_tailor_1.pkl"
    # pickle_file_path = '/home/ninak/mz_val_10_full.pkl'
    #plogger.info(f"Loading data from {pickle_file_path}")
            
    with open(pickle_file_path, "rb") as f:
                data = pickle.load(f)
            
    #df = data["dataframe"] don't have this column
    sequence_arr = data['sequence']
    # sequence = sequence_arr[0]
    mz_values = data["mz_values"]
    # source_dir = '/home/ninak'
    exp_dir = '/home/ninak/exps'
    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)

    exp_id = generate_uuid()
    plogger = get_logger(exp_id=exp_id, log_path=exp_dir)
    plogger.info(f"Experiment ID: {exp_id}")

    plogger.info(f"sequences shape: {sequence_arr.shape}")
    plogger.info(f"mz_values shape: {mz_values.shape}")
    tensorboard_writer = SummaryWriter(log_dir=f"{exp_dir}")
    plogger.info("TensorBoard writer initialized.")
    loss_values = []

    plogger.info(f"Extracted m/z values of shape {len(mz_values[0])}")  # and dtype {mz_values.dtype}")
        
    sequence_unfiltered = sequence_arr[0]
    sequence = remove_brackets(sequence_unfiltered)
    plogger.info(f"Peptide sequence: {sequence}")
    mz_values_i = mz_values[0]
    encoder_input = torch.tensor(mz_values_i, device=device, dtype=torch.float32)
        # Decoder input (sequence)
    decoder_input = add_random_letters(sequence)
    plogger.info(f"Decoder input sequence: {decoder_input}")

        #data.shape[0]
    for i in range(1000):
        plogger.info(f"Run ID: {i}")
        
        output_distribution = transformer_model(encoder_input, decoder_input)
        #plogger.info(f"Model output distribution: {output_distribution}")
        # Loss function
        loss_fn = nn.CrossEntropyLoss()
        
        shape = output_distribution.shape[1]
        target_position = (shape + 1) // 2
        plogger.info(f"Target position: {target_position}")
        target = torch.tensor(target_position, device=device, dtype=torch.long).unsqueeze(dim=0)
        loss = loss_fn(output_distribution.view(-1, output_distribution.size(-1)), target)
        loss_values.append(loss.item())
        optimizer.zero_grad()
        #print('Loss:', loss.item())
        plogger.info(f"Calculated loss: {loss.item()}")
        tensorboard_writer = SummaryWriter(comment='Loss')
        #tensorboard_writer.add_scalar("Loss/train", loss, i)
        loss.backward()
        optimizer.step()
        plogger.info("Completed backpropagation and optimization step.")
    plogger.info(f"Model final output distribution: {output_distribution}")
    for name, param in transformer_model.named_parameters():
        print(f"Parameter name: {name}")
        if param is not None:
            print('exists')

