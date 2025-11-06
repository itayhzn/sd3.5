"""
Stream, crop, and resize images from an HF dataset.

- Streams the dataset (no full local download).
- Accepts any dataset where at least one column is a PIL image (Feature: Image).
- Discards images with min(height, width) < size.
- Center-crops to square, resizes to size x size, and saves as JPEG/PNG/WebP.
- For each image, also writes a sibling '<name>.txt' with caption from 'caption.txt'.
- Prints counts of saved and discarded.
"""

import argparse
import os
from typing import Optional

from datasets import load_dataset
from PIL import Image
from tqdm import tqdm
import huggingface_hub as hf

def find_image_field(example) -> Optional[str]:
    for k, v in example.items():
        if isinstance(v, Image.Image):
            return k
    return None

def center_square_crop(img: Image.Image) -> Image.Image:
    w, h = img.size
    m = min(w, h)
    left = (w - m) // 2
    top = (h - m) // 2
    return img.crop((left, top, left + m, top + m))

def _one_line(text: str) -> str:
    return " ".join(str(text).split())

def main():
    hf.login(token=os.getenv("HF_TOKEN"))

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, help="HF dataset repo id, e.g. 'Leonardo6/yfcc15m'")
    parser.add_argument("--split", default="train", help="Dataset split (default: train)")
    parser.add_argument("--out_root", default="./datasets", help="Root output dir (default: ./datasets)")
    parser.add_argument("--size", type=int, default=256, help="Target square size (default: 256)")
    parser.add_argument("--ext", default="jpg", choices=["jpg", "png", "webp"], help="Output format (default: jpg)")
    parser.add_argument("--max_items", type=int, default=0, help="Stop after N items (0 = all)")
    args = parser.parse_args()

    out_dir = os.path.join(args.out_root, args.dataset.replace("/", "__"))
    os.makedirs(out_dir, exist_ok=True)

    ds_iter = load_dataset(args.dataset, split=args.split, streaming=True)

    saved = 0
    discarded_small = 0
    discarded_errors = 0
    checked_image_field = False
    image_field = None

    pbar = tqdm(ds_iter, desc=f"Streaming {args.dataset}:{args.split}", unit="img")
    for idx, example in enumerate(pbar):
        try:
            if not checked_image_field:
                image_field = find_image_field(example)
                checked_image_field = True
                if image_field is None:
                    raise RuntimeError("No Image column (PIL.Image) found in this dataset.")

            img = example[image_field]
            if not isinstance(img, Image.Image):
                img = Image.fromarray(img)

            w, h = img.size
            if w < args.size or h < args.size:
                discarded_small += 1
                continue

            img = center_square_crop(img)

            if img.mode not in ("RGB", "L"):
                img = img.convert("RGB")
            elif img.mode == "L":
                img = img.convert("RGB")

            if img.size != (args.size, args.size):
                img = img.resize((args.size, args.size), Image.Resampling.LANCZOS)

            key = example.get("__key__", f"{idx:09d}")
            img_name = f"{key}.{args.ext}"
            img_path = os.path.join(out_dir, img_name)

            save_kwargs = {}
            if args.ext == "jpg":
                save_kwargs = dict(quality=95, optimize=True)
            img.save(img_path, **save_kwargs)

            # Write caption file from 'caption.txt' column (empty if missing)
            caption = _one_line(example.get("caption.txt", ""))
            with open(os.path.join(out_dir, f"{key}.txt"), "w", encoding="utf-8") as ftxt:
                ftxt.write(caption + "\n")

            saved += 1

        except Exception:
            discarded_errors += 1

        if args.max_items and (idx + 1) >= args.max_items:
            break

        pbar.set_postfix(saved=saved, small=discarded_small, errors=discarded_errors)

    print("\n=== Summary ===")
    print(f"Dataset:        {args.dataset}:{args.split}")
    print(f"Output dir:     {out_dir}")
    print(f"Saved images:   {saved}")
    print(f"Discarded small:{discarded_small}")
    print(f"Discarded error:{discarded_errors}")
    print(f"Total processed:{saved + discarded_small + discarded_errors}")

if __name__ == "__main__":
    main()