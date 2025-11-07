# query_index.py
import numpy as np
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel
import faiss

class RealIndex:
    def __init__(self, index_dir: str, clip_model_name="openai/clip-vit-large-patch14", device="cuda"):
        self.device = device
        self.clip = CLIPModel.from_pretrained(clip_model_name).to(device).eval()
        self.proc = CLIPProcessor.from_pretrained(clip_model_name)

        self.img_index = faiss.read_index(f"{index_dir}/img.index")
        self.paths = np.load(f"{index_dir}/paths.npy", allow_pickle=True)

        # Optional text index
        try:
            self.text_index = faiss.read_index(f"{index_dir}/text.index")
            self.captions = np.load(f"{index_dir}/captions.npy", allow_pickle=True)
        except:
            self.text_index = None
            self.captions = None

    @torch.no_grad()
    def embed_image(self, img: Image.Image):
        inputs = self.proc(images=img, return_tensors="pt").to(self.device)
        z = self.clip.get_image_features(**inputs)
        z = z / z.norm(dim=-1, keepdim=True)
        return z.cpu().numpy().astype("float32")  # (1, D)

    @torch.no_grad()
    def embed_text(self, prompt: str):
        inputs = self.proc(text=[prompt], return_tensors="pt").to(self.device)
        z = self.clip.get_text_features(**inputs)
        z = z / z.norm(dim=-1, keepdim=True)
        return z.cpu().numpy().astype("float32")  # (1, D)

    def nearest_images(self, img: Image.Image, k=5):
        z = self.embed_image(img)
        D, I = self.img_index.search(z, k)
        return float(D[0, 0]), I[0, 0]  # distances (cosine sims), indices

    def prompt_conditioned(self, prompt: str, k=50, topn=5, w_img=0.5, w_txt=0.5):
        """
        Retrieve top-k by text->image similarity, then re-rank (optional) by a mix of text and (when available) 
        your query image similarity.
        If you only have a prompt (no query image), this returns the most prompt-aligned real images.
        """
        if self.text_index is None:
            raise ValueError("No text index available.")

        t = self.embed_text(prompt)
        D, I = self.text_index.search(t, k)     # highest inner product = most similar
        idx = I[0]
        sims_text = D[0]                         # cosine sims in [-1, 1]

        # Return top-N ranked purely by text similarity (or plug in a fusion you prefer)
        order = np.argsort(-sims_text)[:topn]
        return sims_text[order], idx[order]