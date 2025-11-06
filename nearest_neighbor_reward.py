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
        sim = float(D[0][0])                 # cosine sim ∈ [-1,1]
        reward = float(torch.exp(self.alpha * torch.tensor(sim)).clamp(max=1e6))  # map up; keep sane cap
        return reward
    
# reward_prompt_aware.py
import numpy as np
import torch

class PromptAwareNearestNeighborReward:
    """
    Reward = sigmoid( a * sim_text_image ) * exp( b * sim_to_nearest_real )
    a,b: scalars to trade off prompt fidelity vs realism.
    """
    def __init__(self, index_dir: str, a: float = 5.0, b: float = 10.0, device: str = "cuda"):
        from query_index import RealIndex
        self.idx = RealIndex(index_dir, device=device)
        if self.idx.text_index is None:
            raise ValueError("Index does not include captions/text embeddings.")
        self.a = a
        self.b = b
        self.device = device

    def __call__(self, prompt: str, img) -> float:
        # text-image alignment
        t = self.idx.embed_text(prompt)    # (1, D)
        img_z = self.idx.embed_image(img)  # (1, D)
        txt_sim = float((t @ img_z.T)[0,0])   # cosine since both normalized

        # realism proximity via nearest neighbor
        D, _ = self.idx.img_index.search(img_z, 1)
        real_sim = float(D[0][0])

        # map to (0,1]-ish and combine
        align = torch.sigmoid(self.a * torch.tensor(txt_sim))  # [0,1]
        real  = torch.exp(self.b * torch.tensor(real_sim))     # grows with similarity
        reward = float((align * real).clamp(max=1e6))
        return reward