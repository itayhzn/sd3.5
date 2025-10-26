# mock_scorer.py
# Deterministic, fast, no external deps beyond PIL/numpy
import numpy as np
from PIL import Image, ImageFilter

class MockScorer:
    """
    Returns a scalar reward from a PIL Image.
    Default = "sharpness + contrast" (both normalized to [0,1]).
    Deterministic: given the same image, same score.
    """
    def __init__(self, mode: str = "sharp_contrast"):
        assert mode in ("sharp_contrast", "brightness", "entropy")
        self.mode = mode

    def __call__(self, img: Image.Image) -> float:
        if self.mode == "brightness":
            arr = np.asarray(img.convert("L"), dtype=np.float32) / 255.0
            return float(arr.mean())

        if self.mode == "entropy":
            # 256-bin entropy on grayscale
            arr = np.asarray(img.convert("L"), dtype=np.uint8)
            hist, _ = np.histogram(arr, bins=256, range=(0, 255), density=True)
            hist = hist + 1e-12
            ent = -np.sum(hist * np.log(hist))
            # normalize by log(256)
            return float(ent / np.log(256.0))

        # default: sharpness + contrast (both in [0,1] approx)
        gray = img.convert("L")
        # approximate sharpness via Laplacian-like filter response (using PIL edge filter as proxy)
        edges = gray.filter(ImageFilter.FIND_EDGES)
        e = np.asarray(edges, dtype=np.float32) / 255.0
        sharp = float(e.mean())                   # higher if more edges

        arr = np.asarray(gray, dtype=np.float32) / 255.0
        contrast = float(arr.std() * 2.0)        # rough normalization
        contrast = max(0.0, min(1.0, contrast))

        # blend & clamp
        score = 0.6 * sharp + 0.4 * contrast
        return max(0.0, min(1.0, score))