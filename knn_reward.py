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
        cossim = float(D[0].mean())                 # cosine sim ∈ [-1,1]
        theta = np.arccos(np.clip(cossim, -1.0, 1.0))  # angle in [0, π]
        sim = 1.0 - (theta / np.pi)              # normalize to [0, 1]
        sim = np.exp(3 * sim) - 1.0              # sharpen
        sim = np.clip(sim, 0.0, 1.0)              # ensure sim is in [0, 1]
        return sim