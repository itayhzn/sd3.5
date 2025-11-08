# build_index.py
import os, json, glob, io
from typing import List, Optional, Tuple, Iterable
import numpy as np
from PIL import Image, ImageFile
from tqdm import tqdm
import re
import requests
import torch
from transformers import CLIPProcessor, CLIPModel
import faiss
import argparse

ImageFile.LOAD_TRUNCATED_IMAGES = True

# ----------------------------- disk dataset (existing) -----------------------------

def _iter_dataset(
    roots: List[str],
    img_exts=(".jpg", ".jpeg", ".png", ".webp"),
) -> List[Tuple[str, Optional[str]]]:
    """Return [(image_path, caption_or_None), ...] from local folders."""
    items = []
    for root in roots:
        for p in glob.glob(os.path.join(root, "**", "*"), recursive=True):
            if p.lower().endswith(img_exts):
                caption = None
                stem, _ = os.path.splitext(p)
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
                        # noqa: E701 keep compact
                            j = json.load(f)
                        caption = j.get("caption") or j.get("text")
                    except:
                        pass
                items.append((p, caption))
    return items

# ----------------------------- TSV streaming --------------------------------------

def _iter_tsv_urls(tsv_path: str) -> Iterable[Tuple[str, Optional[str]]]:
    """
    Yields (url, caption_or_None) from a TSV with the format:
      TsvHttpData-1.0
      <url>\t<id>\t<hash>
    Lines with fewer than 1 column after the header are skipped.
    """
    with open(tsv_path, "r", encoding="utf-8") as f:
        first = f.readline()
        # header often 'TsvHttpData-1.0'; tolerate missing/extra header
        if not first or not first.strip().startswith("TsvHttpData"):
            # first line may already be a data row
            parts = first.rstrip("\n").split("\t")
            if parts and parts[0].startswith("http"):
                yield (parts[0], None)
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if not parts:
                continue
            url = parts[0].strip()
            if url.startswith("http"):
                yield (url, None)

def _download_image(url: str, timeout: float = 10.0) -> Optional[Image.Image]:
    try:
        r = requests.get(url, timeout=timeout, stream=True)
        r.raise_for_status()
        b = r.content
        img = Image.open(io.BytesIO(b)).convert("RGB")
        return img
    except Exception:
        return None

# ----------------------------- CLIP helpers ---------------------------------------

@torch.no_grad()
def _embed_pil_batch(model, proc, pil_images, device="cuda"):
    inputs = proc(images=pil_images, return_tensors="pt", padding=True).to(device)
    feats = model.get_image_features(**inputs)  # (B, D)
    feats = torch.nn.functional.normalize(feats, dim=-1)
    return feats

@torch.no_grad()
def _embed_text_batch(model, proc, texts, device="cuda"):
    cleaned = []
    for t in texts:
        t = "" if t is None else str(t)
        t = re.sub(r"\s+", " ", t).strip()
        if t:
            cleaned.append(t)
        else:
            cleaned.append("")  # keep alignment
    inputs = proc(
        text=cleaned,
        padding=True,
        truncation=True,
        max_length=proc.tokenizer.model_max_length,
        return_tensors="pt",
    )
    for k in inputs:
        inputs[k] = inputs[k].to(device)
    feats = model.get_text_features(**inputs)
    feats = torch.nn.functional.normalize(feats, dim=-1)
    return feats

# ----------------------------- FAISS builders -------------------------------------

def _ensure_ip_index(dim: int):
    # Cosine via inner product on unit vectors
    return faiss.IndexFlatIP(dim)

def _append_to_index(index: faiss.Index, feats: torch.Tensor):
    arr = feats.detach().cpu().to(torch.float32).numpy()
    index.add(arr)

# ----------------------------- Main builders --------------------------------------

def build_faiss_from_dirs(
    dataset_dirs: List[str],
    out_dir: str,
    clip_model_name: str = "openai/clip-vit-large-patch14",
    device: str = "cuda",
    build_text_index: bool = True,
    batch_size: int = 64,
):
    os.makedirs(out_dir, exist_ok=True)

    items = _iter_dataset(dataset_dirs)
    if not items:
        raise ValueError("No images found in dataset_dirs.")
    img_paths, captions = zip(*items)

    # Load CLIP
    model = CLIPModel.from_pretrained(clip_model_name).to(device).eval()
    proc = CLIPProcessor.from_pretrained(clip_model_name)

    # Embed images in batches to avoid large memory
    img_index = None
    all_paths = []
    for i in tqdm(range(0, len(img_paths), batch_size), desc="Embed images (dirs)"):
        paths_chunk = img_paths[i : i + batch_size]
        pil_batch = [Image.open(p).convert("RGB") for p in paths_chunk]
        feats = _embed_pil_batch(model, proc, pil_batch, device=device)
        if img_index is None:
            img_index = _ensure_ip_index(feats.shape[1])
        _append_to_index(img_index, feats)
        all_paths.extend(paths_chunk)

    faiss.write_index(img_index, os.path.join(out_dir, "img.index"))
    np.save(os.path.join(out_dir, "paths.npy"), np.array(all_paths))

    if build_text_index:
        text_index = None
        all_caps = list(captions)
        for i in tqdm(range(0, len(all_caps), 256), desc="Embed captions"):
            caps_chunk = all_caps[i : i + 256]
            feats = _embed_text_batch(model, proc, caps_chunk, device=device)
            if text_index is None:
                text_index = _ensure_ip_index(feats.shape[1])
            _append_to_index(text_index, feats)
        faiss.write_index(text_index, os.path.join(out_dir, "text.index"))
        np.save(os.path.join(out_dir, "captions.npy"), np.array(all_caps, dtype=object))

    print(f"[OK] Saved index to {out_dir}.")

def build_faiss_from_tsv(
    tsv_path: str,
    out_dir: str,
    clip_model_name: str = "openai/clip-vit-large-patch14",
    device: str = "cuda",
    batch_size: int = 64,
    max_items: int = 0,              # 0 = all
    min_side: int = 0,               # optionally skip too-small images
    build_text_index: bool = False,  # TSV has no captions; keep off by default
    timeout: float = 10.0,
    print_every: int = 100,          # how often to print progress summaries
):
    """
    Streams a TSV of URLs:
      - downloads up to `batch_size` images into memory,
      - embeds, adds to FAISS,
      - discards them and continues.
    Shows real-time progress with flushes.
    """
    os.makedirs(out_dir, exist_ok=True)

    print(f"[INFO] Starting FAISS index build from TSV: {tsv_path}", flush=True)
    print(f"[INFO] Output directory: {out_dir}", flush=True)
    print(f"[INFO] Device: {device} | Batch size: {batch_size}", flush=True)
    print(f"[INFO] CLIP model: {clip_model_name}", flush=True)

    # Load CLIP
    model = CLIPModel.from_pretrained(clip_model_name).to(device).eval()
    proc = CLIPProcessor.from_pretrained(clip_model_name)

    img_index = None
    paths_accum = []
    pil_batch, url_batch = [], []

    n_seen = 0
    n_success = 0
    n_failed = 0
    n_skipped_small = 0

    for url, _ in _iter_tsv_urls(tsv_path):
        if max_items and n_seen >= max_items:
            break

        n_seen += 1
        img = _download_image(url, timeout=timeout)
        if img is None:
            n_failed += 1
            if n_seen % print_every == 0:
                print(f"[{n_seen}] Failed to download {url}", flush=True)
            continue

        if min_side > 0:
            w, h = img.size
            if min(w, h) < min_side:
                n_skipped_small += 1
                continue

        pil_batch.append(img)
        url_batch.append(url)

        # Process when batch is full
        if len(pil_batch) >= batch_size:
            feats = _embed_pil_batch(model, proc, pil_batch, device=device)
            if img_index is None:
                img_index = _ensure_ip_index(feats.shape[1])
            _append_to_index(img_index, feats)
            paths_accum.extend(url_batch)
            n_success += len(pil_batch)

            print(
                f"[PROGRESS] {n_success} images indexed "
                f"({n_seen} seen, {n_failed} failed, {n_skipped_small} too small)",
                flush=True,
            )

            # discard batch to free memory
            pil_batch.clear()
            url_batch.clear()

    # Flush remaining
    if pil_batch:
        feats = _embed_pil_batch(model, proc, pil_batch, device=device)
        if img_index is None:
            img_index = _ensure_ip_index(feats.shape[1])
        _append_to_index(img_index, feats)
        paths_accum.extend(url_batch)
        n_success += len(pil_batch)
        print(f"[FLUSH] Added last {len(pil_batch)} images (total {n_success}).", flush=True)

    if img_index is None:
        raise ValueError("No valid images were embedded from TSV.")

    # Save results
    faiss.write_index(img_index, os.path.join(out_dir, "img.index"))
    np.save(os.path.join(out_dir, "paths.npy"), np.array(paths_accum, dtype=object))

    print("\n=== SUMMARY ===", flush=True)
    print(f"TSV path:         {tsv_path}", flush=True)
    print(f"Images seen:      {n_seen}", flush=True)
    print(f"Images indexed:   {n_success}", flush=True)
    print(f"Images failed:    {n_failed}", flush=True)
    print(f"Too small:        {n_skipped_small}", flush=True)
    print(f"Output directory: {out_dir}", flush=True)
    print(f"✅ Saved FAISS index successfully.", flush=True)

# ----------------------------- CLI -----------------------------------------------

def main():
    p = argparse.ArgumentParser()
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--dataset-dirs", nargs="+", help="One or more folders of images (+optional .txt/.json captions).")
    g.add_argument("--tsv", help="Path to TSV file with 'TsvHttpData-1.0' header and <url> per line.")
    p.add_argument("--out-dir", required=True, help="Where to write FAISS index and metadata.")
    p.add_argument("--clip-model", default="openai/clip-vit-large-patch14")
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--max-items", type=int, default=0, help="TSV mode: cap number of images processed (0=all).")
    p.add_argument("--min-side", type=int, default=0, help="TSV mode: skip images with min(width,height) < min-side.")
    p.add_argument("--build-text-index", action="store_true", help="(Dirs mode) also index captions when present.")
    p.add_argument("--timeout", type=float, default=10.0, help="HTTP timeout per image in seconds (TSV mode).")
    args = p.parse_args()

    if args.tsv:
        build_faiss_from_tsv(
            tsv_path=args.tsv,
            out_dir=args.out_dir,
            clip_model_name=args.clip_model,
            device=args.device,
            batch_size=args.batch_size,
            max_items=args.max_items,
            min_side=args.min_side,
            build_text_index=False,
            timeout=args.timeout,
        )
    else:
        build_faiss_from_dirs(
            dataset_dirs=args.dataset_dirs,
            out_dir=args.out_dir,
            clip_model_name=args.clip_model,
            device=args.device,
            build_text_index=args.build_text_index,
            batch_size=args.batch_size,
        )

if __name__ == "__main__":
    main()