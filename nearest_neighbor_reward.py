# reward_realism.py
import numpy as np
import torch

class NearestNeighborReward:
    """
    Reward = exp(alpha * sim_to_nearest_real)
    where sim is cosine (inner product on unit vectors).
    alpha scales sharpness (try 5.0~15.0).
    """
    def __init__(self, index_dir: str, alpha: float = 10.0, device: str = "cuda"):
        from query_index import RealIndex
        self.idx = RealIndex(index_dir, device=device)
        self.alpha = alpha

    def __call__(self, prompt: str, img) -> float:
        z = self.idx.embed_image(img)        # (1, D)
        D, I = self.idx.img_index.search(z, 1)
        sim = float(D[0, 0])                 # cosine sim ∈ [-1,1]
        
        return sim