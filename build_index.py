# build_index.py
import os, json, glob
from typing import List, Optional, Tuple
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
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
def _batch_embed_texts(model, proc, texts, device="cuda", batch_size=256):
    embs = []
    # Replace None captions with empty string for shape safety
    texts = [t if (t is not None) else "" for t in texts]
    for i in tqdm(range(0, len(texts), batch_size), desc="Embed captions"):
        batch = texts[i:i+batch_size]
        inputs = proc(text=batch, return_tensors="pt", padding=True).to(device)
        feats = model.get_text_features(**inputs)
        feats = feats / feats.norm(dim=-1, keepdim=True)
        embs.append(feats.cpu())
    return torch.cat(embs, dim=0).numpy()  # (N, D)


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
        if cap_emb.shape[1] != D:
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
        # Save captions (aligned)
        np.save(os.path.join(out_dir, "captions.npy"), np.array(captions, dtype=object))

    print(f"Saved index to {out_dir}.")