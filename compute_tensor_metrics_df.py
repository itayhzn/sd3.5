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

def compute_metric_in_stream(base_dir='tensors/outputs/sd3.5_medium', dir_predicate=lambda x: True, tensor_predicate=lambda x, d: x.startswith('x_grad_t='), metric_fns={'mean': lambda x: np.mean(np.array(x))}):
    metrics = []
    for path in tqdm(os.listdir(base_dir)):
        if not dir_predicate(path):
            continue
        tensors = read_tensors_aux(os.path.join(base_dir, path, '000000'), tensor_predicate)
        ks = sorted(list(tensors.keys()))[::-1]
        grad = np.array([tensors[k] for k in ks])
        for k, v in tensors.items():
            d = { 'path': path}
            d['tensor_name'] = ''.join(k.split('=')[:-1])
            d['timestep'] =int(k.split('=')[-1])
            for name, metric_fn in metric_fns.items():
                d[name] = metric_fn(v[np.newaxis, ...])[0]
            metrics.append(d)

    return pd.DataFrame(metrics)


if __name__ == "__main__":
    metric_fns = {
        'saliency_mass': compute_saliency_mass
    }

    dir_predicate = lambda x: 'dataset-02' in x

    df = compute_metric_in_stream(dir_predicate=dir_predicate, metric_fns=metric_fns)
    df.to_csv('tensor_metrics_df.csv', index=False)
