# build_index.py
import os, json, glob
from typing import List, Optional, Tuple
import numpy as np
from PIL import Image
from tqdm import tqdm

import re
import torch
from torch.utils.data import DataLoader

from transformers import CLIPProcessor, CLIPModel
import faiss


def _iter_dataset(
    roots: List[str],
    img_exts=(".jpg", ".jpeg", ".png", ".webp"),
    caption_suffixes=(".txt", ".json"),
) -> List[Tuple[str, Optional[str]]]:
    """
    Walk directories and return [(image_path, caption_text_or_None), ...].
    Caption discovery rules:
      - If image.jpg has image.txt next to it, load that as the caption (first line).
      - If image.json has {"caption": "..."} or {"text": "..."} we read that.
      - Otherwise caption=None.
    """
    items = []
    for root in roots:
        for p in glob.glob(os.path.join(root, "**", "*"), recursive=True):
            if p.lower().endswith(img_exts):
                caption = None
                stem, _ = os.path.splitext(p)
                # Prefer .txt
                txt_path = stem + ".txt"
                json_path = stem + ".json"
                if os.path.exists(txt_path):
                    try:
                        with open(txt_path, "r", encoding="utf-8") as f:
                            caption = f.readline().strip()
                    except:
                        pass
                elif os.path.exists(json_path):
                    try:
                        with open(json_path, "r", encoding="utf-8") as f:
                            j = json.load(f)
                        caption = j.get("caption") or j.get("text")
                    except:
                        pass
                items.append((p, caption))
    return items


@torch.no_grad()
def _batch_embed_images(model, proc, paths, device="cuda", batch_size=64):
    embs = []
    for i in tqdm(range(0, len(paths), batch_size), desc="Embed images"):
        batch = [Image.open(p).convert("RGB") for p in paths[i:i+batch_size]]
        inputs = proc(images=batch, return_tensors="pt", padding=True).to(device)
        feats = model.get_image_features(**inputs)
        feats = feats / feats.norm(dim=-1, keepdim=True)
        embs.append(feats.cpu())
    return torch.cat(embs, dim=0).numpy()  # (N, D)


@torch.no_grad()
def _batch_embed_texts(model, processor, texts, device="cuda", batch_size=256):
    # Clean & filter captions
    cleaned = []
    for t in texts:
        t = "" if t is None else str(t)
        t = re.sub(r"\s+", " ", t).strip()
        if t:  # keep non-empty
            cleaned.append(t)
    if not cleaned:
        # Return empty (0, D) NumPy to keep downstream happy
        D = model.text_projection.out_features
        return np.empty((0, D), dtype=np.float32)

    embs = []
    for i in range(0, len(cleaned), batch_size):
        chunk = cleaned[i:i+batch_size]

        # IMPORTANT: pad + truncate so we can return tensors;
        # limit to CLIP's max (typically 77)
        inputs = processor(
            text=chunk,
            padding=True,                      # <-- changed
            truncation=True,                   # <-- keep
            max_length=processor.tokenizer.model_max_length,
            return_tensors="pt",
        )
        # move to device
        for k in inputs:
            inputs[k] = inputs[k].to(device)

        feats = model.get_text_features(**inputs)         # (B, D)
        feats = torch.nn.functional.normalize(feats, dim=-1)
        embs.append(feats.cpu())

    return torch.cat(embs, dim=0).numpy()                 # <-- return NumPy


def build_faiss_indexes(
    dataset_dirs: List[str],
    out_dir: str,
    clip_model_name: str = "openai/clip-vit-large-patch14",
    device: str = "cuda",
    build_text_index: bool = True,
):
    os.makedirs(out_dir, exist_ok=True)

    # 1) Collect dataset
    items = _iter_dataset(dataset_dirs)
    if not items:
        raise ValueError("No images found.")
    img_paths, captions = zip(*items)
    img_paths = list(img_paths)
    captions = list(captions)

    # 2) Load CLIP
    model = CLIPModel.from_pretrained(clip_model_name).to(device).eval()
    proc  = CLIPProcessor.from_pretrained(clip_model_name)

    # 3) Compute embeddings
    img_emb = _batch_embed_images(model, proc, img_paths, device=device)
    D = img_emb.shape[1]

    if build_text_index:
        cap_emb = _batch_embed_texts(model, proc, captions, device=device)
        if cap_emb.size == 0:
            print("[WARN] No non-empty captions found; skipping text index.")
            build_text_index = False
        elif cap_emb.shape[1] != D:
            raise ValueError("Text and image embedding dims differ.")

    # 4) Build FAISS (cosine via inner product on unit vectors)
    img_index = faiss.IndexFlatIP(D)
    img_index.add(img_emb.astype(np.float32))
    faiss.write_index(img_index, os.path.join(out_dir, "img.index"))
    np.save(os.path.join(out_dir, "paths.npy"), np.array(img_paths))

    if build_text_index:
        text_index = faiss.IndexFlatIP(D)
        text_index.add(cap_emb.astype(np.float32))
        faiss.write_index(text_index, os.path.join(out_dir, "text.index"))
        np.save(os.path.join(out_dir, "captions.npy"), np.array(captions, dtype=object))

    print(f"Saved index to {out_dir}.")