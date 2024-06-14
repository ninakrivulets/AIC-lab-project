import csv
import datetime
import os
import math
import platform
import sys

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from Bio import pairwise2
from torch.utils.tensorboard import SummaryWriter

os_name = platform.system()
os_version = platform.release()

torch.set_num_threads(1)

torch_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print("hello world!")

print(f"Operating System: {os_name}")
print(f"OS Version:  {os_version}")
print(f"Python Version: {sys.version}")
print(f"Numpy Version: {np.__version__}")
print(f"PyTorch Version: {torch.__version__}")

