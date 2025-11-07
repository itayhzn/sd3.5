# reward_realism.py
import numpy as np
import torch

class KNNReward:
    def __init__(self, index_dir: str, k: int = 1, device: str = "cuda"):
        from query_index import RealIndex
        self.idx = RealIndex(index_dir, device=device)
        self.k = k

    def __call__(self, prompt: str, img) -> float:
        z = self.idx.embed_image(img)        # (1, D)
        D, I = self.idx.img_index.search(z, self.k)
        sim = float(D[0].mean())                 # cosine sim ∈ [-1,1]
        
        return sim