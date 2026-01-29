"""
config.py

Configurations for data_io

"""
from __future__ import absolute_import

import os
import sys
import numpy as np
import torch
import torch.utils.data

h_dtype = np.float32
h_dtype_str = '<f4'
d_dtype = torch.float32
std_floor = 0.00000001
mean_std_i_file = 'mean_std_input.bin'
mean_std_o_file = 'mean_std_output.bin'
data_len_file = 'utt_length.dic'
f0_unvoiced_dic = {'.f0' : 0}
data_seq_min_length = 40
default_loader_conf = {'batch_size':1, 'shuffle':False, 'num_workers':0}
