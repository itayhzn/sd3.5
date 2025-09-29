from datetime import datetime
import sys
import torch
import os
import numpy as np

def save_tensors(save_tensors_dir, tensors_dict):
    r"""
    Save tensors to disk for debugging purposes.
    """
    # mkdir if not exists
    if not os.path.exists(save_tensors_dir):
        os.makedirs(save_tensors_dir)

    print(f'======= Saving tensors to {save_tensors_dir}')
    for name, tensor in tensors_dict.items():
        if isinstance(tensor, torch.Tensor):
            print(f'\tSaving tensor {name} to {os.path.join(save_tensors_dir, name)}')
            torch.save(tensor, os.path.join(save_tensors_dir, f'{name}.pt'))
        else:
            print(f'{name} is not a tensor, skipping save. Type: {type(tensor)}')
    print(f'======= Saved tensors to {save_tensors_dir}')
