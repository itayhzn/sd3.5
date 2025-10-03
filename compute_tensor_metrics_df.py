import sys
import torch
import numpy as np
import os
from tqdm import tqdm
import pandas as pd

def read_tensors_aux(dir, pt_name_predicate=None):
    """ recursively read tensors from a directory """
    if pt_name_predicate is None:
        pt_name_predicate = lambda x, y: True
    tensors = {}
    for path in os.listdir(dir):
        if path.endswith('.pt'):
            key = path[:-3]
            if pt_name_predicate is None or pt_name_predicate(key, dir):
                try:
                    tensors[key] = torch.load(os.path.join(dir, path))
                except Exception as e:
                    print(f"Error loading {path}: {e}")
        elif os.path.isdir(os.path.join(dir, path)):
            key = path.split('_')[1] if path.startswith('timestep_') else path 
            tensors[key] = read_tensors_aux(os.path.join(dir, path), pt_name_predicate)
        else:
            print(f"Skipping {path}, not a tensor or directory")
    return tensors

def compute_saliency_mass(grads):
    """
        grads: tensor of shape (T, W, H) or list of tensors of shape (W, H)
        Returns: tensor of shape (T,) with the saliency mass for each timestep
    """
    if isinstance(grads, list):
        grads = torch.tensor(grads)
    # grads = torch.max(grads, axis=1)  # max pooling over channels
    # W,H = grads.shape[1], grads.shape[2]
    return torch.sum(torch.abs(grads), axis=(1, 2)).cpu().numpy()

def compute_l2_norm(grads):
    """
        grads: tensor of shape (T, W, H) or list of tensors of shape (W, H)
        Returns: tensor of shape (T,) with the l2 norm for each timestep
    """
    if isinstance(grads, list):
        grads = torch.tensor(grads)
    return torch.norm(grads.view(grads.shape[0], -1), dim=1).cpu().numpy()

def compute_l1_norm(grads):
    """
        grads: tensor of shape (T, W, H) or list of tensors of shape (W, H)
        Returns: tensor of shape (T,) with the l1 norm for each timestep
    """
    if isinstance(grads, list):
        grads = torch.tensor(grads)
    return torch.norm(grads.view(grads.shape[0], -1), p=1, dim=1).cpu().numpy()

def compute_max_norm(grads):
    """
        grads: tensor of shape (T, W, H) or list of tensors of shape (W, H)
        Returns: tensor of shape (T,) with the max norm for each timestep
    """
    if isinstance(grads, list):
        grads = torch.tensor(grads)
    return torch.max(grads.view(grads.shape[0], -1), dim=1).values.cpu().numpy()

def compute_mean_norm(grads):
    """
        grads: tensor of shape (T, W, H) or list of tensors of shape (W, H)
        Returns: tensor of shape (T,) with the mean norm for each timestep
    """
    if isinstance(grads, list):
        grads = torch.tensor(grads)
    return torch.mean(grads.view(grads.shape[0], -1), dim=1).cpu().numpy()

def compute_var_norm(grads):
    """
        grads: tensor of shape (T, W, H) or list of tensors of shape (W, H)
        Returns: tensor of shape (T,) with the var norm for each timestep
    """
    if isinstance(grads, list):
        grads = torch.tensor(grads)
    return torch.var(grads.view(grads.shape[0], -1), dim=1).cpu().numpy()

def compute_var_on_diff(grads):
    """
        grads: tensor of shape (T, W, H) or list of tensors of shape (W, H)
        Returns: tensor of shape (T-1,) with the var norm on the diff for each timestep
    """
    if isinstance(grads, list):
        grads = torch.tensor(grads) # (T, W, H)
    diffs = torch.diff(grads, dim=0) # (T-1, W, H)
    return torch.var(diffs.view(diffs.shape[0], -1), dim=1).cpu().numpy()

def compute_metric_in_stream(base_dir='tensors/outputs/sd3.5_medium', dir_predicate=lambda x: True, tensor_predicate=lambda x, d: x.startswith('x_grad_t='), metric_fns={'mean': lambda x: np.mean(np.array(x))}):
    metrics = []
    for path in tqdm(os.listdir(base_dir)):
        if not dir_predicate(path):
            continue
        tensors = read_tensors_aux(os.path.join(base_dir, path, '000000'), tensor_predicate)
        ks = sorted(list(tensors.keys()))[::-1]
        grad = torch.stack([tensors[k] for k in ks])
        for metric_name, metric_fn in metric_fns.items():
            ms = metric_fn(grad) # (T,)
            for k, m in zip(ks.tolist(), ms.tolist()):
                d = { 'path': path}
                d['tensor_name'] = ''.join(k.split('=')[:-1])
                d['timestep'] =int(k.split('=')[-1])
                d[metric_name] = float(m)
                metrics.append(d)
        sys.stdout.flush()
        
    return pd.DataFrame(metrics)


if __name__ == "__main__":
    metric_fns = {
        'saliency_mass': compute_saliency_mass,
        'l2': compute_l2_norm,
        'l1': compute_l1_norm,
        'max': compute_max_norm,
        'mean': compute_mean_norm,
        'var': compute_var_norm,
        'var_on_diff': compute_var_on_diff,
    }

    dir_predicate = lambda x: 'dataset-03' in x

    df = compute_metric_in_stream(dir_predicate=dir_predicate, metric_fns=metric_fns)
    df.to_csv('tensor_metrics_df.csv', index=False)
