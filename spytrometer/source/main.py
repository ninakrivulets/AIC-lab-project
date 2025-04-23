import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import pandas as pd
import os
import glob
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
from model_with_pos_encoding import Transformer
from model_with_pos_encoding import generate_uuid
from model_with_pos_encoding import get_logger
# from model_with_pos_encoding import AAEmbedding
from model_with_pos_encoding import MultiTargetLoss
# from model_with_pos_encoding import remove_brackets
# from model_with_pos_encoding import add_random_letters

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
    device = torch.device("cuda:"+str(GPU_ID) if torch.cuda.is_available() else "cpu")
    plogger.info(f"Torch version: {torch.__version__}")
    plogger.info(f"Torch device: {device}")
    num_heads = 4
    num_layers = 3
    model_dim = 32
    kernel_stride = 11 # Used in proteome embedding to reduce the proteom length.
    plogger.info(f"Model num head: {num_heads}")
    plogger.info(f"Model num layer: {num_layers}")
    plogger.info(f"Model dim : {model_dim}")
    plogger.info(f"Kernel stride in proteome embedding: {kernel_stride}")
    if model_dim % num_heads > 0 :
        plogger.info(f"Model dim is not dividable with head num. Terminating with error.")
        exit(-1)

    transformer_model = Transformer(device=device, num_heads=num_heads, num_layers=num_layers, d_model=model_dim, kernel_stride=kernel_stride)
    optimizer = torch.optim.Adam(transformer_model.parameters(), lr=1e-3)
    optimizer.zero_grad()            

    loss_fn = MultiTargetLoss().to(device)
    plogger.info(f"The model:\n{transformer_model}")
    plogger.info(f"The optimizer:\n{optimizer}")
    plogger.info(f"The loss function :\n{loss_fn}")

    data_dir = "/blob/dda/PXD028806/"
    pickle_file_path = data_dir + "training_data/"
    training_data = glob.glob(pickle_file_path + "*.pkl")
    plogger.info(f"The training data: {data_dir}")
    plogger.info(f"training data files: {training_data}")

    training_step = 0
    max_training_step = 1e6  # This could be 1e9 or something like this
    batch_size = 8

    # Read the proteome
    with open('/home/data/Fasta/uniprot-proteome_UP000005640+reviewed_yes.pkl', 'rb') as f:  # 'rb' = read binary
        data = pickle.load(f)
        proteome_seq = data['long_sequence']
        proteome_pos = data['position_dict']
    plogger.info(f"Proteome length: {len(proteome_seq)}")
    plogger.info(f"Number of proteins: {len(proteome_pos)}")

    with open('/home/ninak/protein_seq_dict.pkl', 'rb') as f:  # 'rb' = read binary
        protein_id_seq_dict = pickle.load(f)
    plogger.info(f"Loaded dictionary with {len(protein_id_seq_dict)} entries.")

    while True:
        plogger.info(f"New epoch has started!")
        # Iterate over the training data files
        for training_data_file in training_data:
            
            with open(training_data_file, "rb") as f:
                data = pickle.load(f)
            plogger.info(f"Reading training data file:  {training_data_file}, Data read: {len(data)}")
            file_loss = 0.0

            # random.shuffle(data)
            data.sample(frac=1).reset_index(drop=True)  
            # Iterate over the training data
            for index, data_item in data.iterrows():

                peaks_mz = data_item.mz_values
                sequence = data_item.sequence
                sequence_ids = data_item.protein_ids
                #targets = []
                for prot_id, pos_dict in sequence_ids.items():
                    if prot_id in proteome_pos:
                        protein_position = proteome_pos[prot_id]
                        peptide_positions = list(range((protein_position+sequence_ids[prot_id]['start'])// kernel_stride, (protein_position+sequence_ids[prot_id]['end'])//kernel_stride))
                        # print("the position:", proteome_pos[prot_id])
                        # print("peptide id start:", sequence_ids[prot_id]['start'])
                        # print("peptide id start:", sequence_ids[prot_id]['end'])
                        # print("peptide_position:",peptide_position )
                        targets += peptide_positions
                # print(targets)
                if len(targets) == 0:
                    continue

                # Forward pass                
                output_distribution = transformer_model(peaks_mz, proteome_seq).squeeze(0)
                # plogger.info(f"Model output distribution: {output_distribution}")
            
                # Loss function
                loss = loss_fn(output_distribution, targets)  # Targets is a list of multiple targets, its like a multi-label classification

                loss_to_report = loss.detach().data.cpu().numpy()
                file_loss += loss_to_report

                training_step += 1

                if np.isnan(loss_to_report):
                    plogger.info(f"loss: {loss_to_report}")
                    sys.exit(f"Loss function is nan - {loss_to_report}!")

                tensorboard_writer.add_scalar("Loss/train", loss_to_report, training_step)
                loss.backward()
                if training_step % batch_size == 0: 
                    # print(transformer_model.decoder_embedding.embedding_layer.weight.grad.shape)
                    # print(transformer_model.decoder_embedding.embedding_layer.weight.grad)
                    # print(transformer_model.decoder_embedding.embedding_layer.weight)
                    optimizer.step()
                    optimizer.zero_grad()            
                    # print(transformer_model.decoder_embedding.embedding_layer.weight)
                    # plogger.info(f"Loss on last batch: {batch_loss/batch_size}")
                    batch_loss = 0.0
                if training_step > max_training_step:
                    plogger.info("Max training step reached. Terminating normally. Bye.")
                    exit(1)
            plogger.info(f"Mean loss on the file:  {file_loss/len(data)}")
            
                
plogger.info("Done. Bye.")