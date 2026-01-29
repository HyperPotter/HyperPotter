
from __future__ import absolute_import

import os
import sys
import numpy as np

import torch
import torch.utils.data
import torch.utils.data.sampler as torch_sampler

import core_scripts.math_tools.random_tools as nii_rand_tk
import core_scripts.other_tools.display as nii_warn

g_str_sampler_bsbl = 'block_shuffle_by_length'

class SamplerBlockShuffleByLen(torch_sampler.Sampler):
    def __init__(self, buf_dataseq_length, batch_size):
        if batch_size == 1:
            mes = "Sampler block shuffle by length requires batch-size>1"
            nii_warn.f_die(mes)
        self.m_block_size = batch_size * 4
        self.m_idx = np.argsort(buf_dataseq_length)
        return
    
    def __iter__(self):
        tmp_list = list(self.m_idx.copy())
        nii_rand_tk.f_shuffle_in_block_inplace(tmp_list, self.m_block_size)
        nii_rand_tk.f_shuffle_blocks_inplace(tmp_list, self.m_block_size)
        return iter(tmp_list)


    def __len__(self):
        return len(self.m_idx)

if __name__ == "__main__":
    print("Definition of customized_sampler")
