import pathlib as pl
import time
import numpy as np
import torch
import torch.nn as nn
import mne as mne
from Model import tf_model, tf_data

n = 100
perm = torch.randperm(n)
print(perm)