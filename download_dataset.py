"""
download_stanford_dogs.py
--------------------------
Download the Hugging Face dataset `ksaml/Stanford_dogs`
and save all images to ./datasets/ksaml/Stanford_dogs/.

Each file is saved as: <label>_<uuid>.<ext>

Example:
  python download_stanford_dogs.py
"""

import os
import uuid
from datasets import load_dataset
from PIL import Image

def main():
    dataset_name = "ksaml/Stanford_dogs"
    out_dir = os.path.join("datasets", dataset_name.replace("/", os.sep))
    os.makedirs(out_dir, exist_ok=True)

    # Stream dataset to avoid full download
    ds_iter = load_dataset(dataset_name, split="train", streaming=True)

    count, errors = 0, 0
    for example in ds_iter:
        try:
            img = example["image"]
            label = str(example["label"])
            ext = "jpg"

            filename = f"{label}_{uuid.uuid4().hex[:8]}.{ext}"
            out_path = os.path.join(out_dir, filename)

            # Ensure RGB mode
            if not isinstance(img, Image.Image):
                img = Image.fromarray(img)
            if img.mode not in ("RGB", "L"):
                img = img.convert("RGB")
            elif img.mode == "L":
                img = img.convert("RGB")

            img.save(out_path, format="JPEG", quality=95)
            count += 1
        except Exception as e:
            errors += 1
            print(f"[WARN] Skipped one image: {e}")

    print(f"\n✅ Done.")
    print(f"Saved images: {count}")
    print(f"Skipped (errors): {errors}")
    print(f"Output directory: {out_dir}")

if __name__ == "__main__":
    main()
