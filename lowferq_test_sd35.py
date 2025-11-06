#!/usr/bin/env python
"""
Low-frequency hypothesis test for SD3.5 initial noise.

Hypothesis:
  Only the low frequencies in the initial noise determine the final image.
  Equivalently: if two initial noises share the same low frequencies but have different high
  frequencies, the resulting image will be (nearly) the same.

This script plugs into the provided SD3.5 reference code (sd3_infer.py, sd3_impls.py) and:
  1) Constructs two noises that share the same low-frequency component but differ in high-freqs.
  2) Samples images with each.
  3) Compares and exports results (images + simple metrics).

Example:
  python lowfreq_test_sd35.py \
      --prompt "a ginger cat, 35mm photo, natural light" \
      --model models/sd3.5_medium.safetensors \
      --clip_g models/clip_g.safetensors \
      --clip_l models/clip_l.safetensors \
      --t5 models/t5xxl.safetensors \
      --vae models/sd3_vae.safetensors \
      --width 1024 --height 1024 --steps 30 --cfg 5 \
      --sampler dpmpp_2m \
      --seed_low 1234 --seed_high_a 111 --seed_high_b 222 \
      --cutoff_frac 0.10 \
      --out_dir outputs/lowfreq_test
"""

from pathlib import Path
import os
import json
import math
import argparse
import numpy as np
from PIL import Image
import torch

import sd3_infer as sd3i
import sd3_impls as sd3_impls


def fft_low_high_split(x: torch.Tensor, cutoff_frac: float):
    """
    Split spatial frequencies of a noise tensor into low and high components.
    x: [B, C, H, W], real tensor
    cutoff_frac: (0, 0.5], fraction of Nyquist radius to keep as "low".
    Returns: (low, high) in the same dtype/device as x.
    """
    assert x.ndim == 4, f"expected [B,C,H,W], got {tuple(x.shape)}"
    B, C, H, W = x.shape
    assert 0.0 < cutoff_frac <= 0.5, "cutoff_frac must be in (0, 0.5]"

    dev = x.device
    x32 = x.to(torch.float32)

    X = torch.fft.fft2(x32, dim=(-2, -1))
    X = torch.fft.fftshift(X, dim=(-2, -1))

    yy = torch.linspace(-0.5, 0.5, steps=H, device=dev).reshape(H, 1).expand(H, W)
    xx = torch.linspace(-0.5, 0.5, steps=W, device=dev).reshape(1, W).expand(H, W)
    rr = torch.sqrt(xx * xx + yy * yy)  # normalized radius

    mask = (rr <= cutoff_frac).to(torch.float32).view(1, 1, H, W)

    X_low = X * mask
    X_high = X * (1.0 - mask)

    X_low = torch.fft.ifftshift(X_low, dim=(-2, -1))
    X_high = torch.fft.ifftshift(X_high, dim=(-2, -1))

    low = torch.fft.ifft2(X_low, dim=(-2, -1)).real.to(x.dtype)
    high = torch.fft.ifft2(X_high, dim=(-2, -1)).real.to(x.dtype)
    return low, high


def compose_noise_with_shared_low(low_shared: torch.Tensor, noise_src: torch.Tensor, cutoff_frac: float):
    """Replace the low-frequencies of noise_src with low_shared; keep noise_src's high-frequencies."""
    _, high = fft_low_high_split(noise_src, cutoff_frac)
    return (low_shared + high).to(noise_src.dtype)


@torch.inference_mode()
def sample_with_custom_noise(
    infer: sd3i.SD3Inferencer,
    latent: torch.Tensor,
    noise: torch.Tensor,
    conditioning,
    neg_cond,
    steps: int,
    cfg_scale: float,
    sampler: str,
    controlnet_cond=None,
    denoise: float = 1.0,
    skip_layer_config=None,
    save_tensors_path=None,
) -> torch.Tensor:
    """
    Equivalent to SD3Inferencer.do_sampling but with externally-specified initial noise.
    Returns latent in VAE space (before decode).
    """
    skip_layer_config = skip_layer_config or {}
    latent = latent.half().cuda()
    infer.sd3.model = infer.sd3.model.cuda()

    noise = noise.to(device=latent.device, dtype=torch.float16 if latent.dtype==torch.float16 else latent.dtype)

    sigmas = infer.get_sigmas(infer.sd3.model.model_sampling, steps).cuda()
    sigmas = sigmas[int(steps * (1 - denoise)) :]
    conditioning = infer.fix_cond(conditioning)
    neg_cond = infer.fix_cond(neg_cond)

    extra_args = {
        "cond": conditioning,
        "uncond": neg_cond,
        "cond_scale": cfg_scale,
        "controlnet_cond": controlnet_cond,
        "save_tensors_path": save_tensors_path,
    }

    noise_scaled = infer.sd3.model.model_sampling.noise_scaling(
        sigmas[0], noise, latent, infer.max_denoise(sigmas)
    )
    sample_fn = getattr(sd3_impls, f"sample_{sampler}")
    denoiser = (
        sd3_impls.SkipLayerCFGDenoiser if skip_layer_config.get("scale", 0) > 0 else sd3_impls.CFGDenoiser
    )
    latent = sample_fn(
        denoiser(infer.sd3.model, steps, skip_layer_config),
        noise_scaled,
        sigmas,
        save_tensors_path=save_tensors_path,
        extra_args=extra_args
    )
    latent = sd3i.SD3LatentFormat().process_out(latent)
    return latent


def mse(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    return float(np.mean((a - b) ** 2))


def psnr(mse_val: float, max_val: float = 255.0) -> float:
    if mse_val <= 1e-12:
        return float("inf")
    return 20.0 * math.log10(max_val) - 10.0 * math.log10(mse_val)


def save_image(arr: np.ndarray, path: Path):
    img = Image.fromarray(arr.astype(np.uint8))
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path)


def run_experiment(
    prompt: str,
    model: str,
    clip_g: str,
    clip_l: str,
    t5: str,
    vae: str,
    width: int,
    height: int,
    steps: int,
    cfg: float,
    sampler: str,
    seed_low: int,
    seed_high_a: int,
    seed_high_b: int,
    cutoff_frac: float,
    out_dir: Path,
    text_encoder_device: str = "cpu",
    verbose: bool = False,
):
    out_dir.mkdir(parents=True, exist_ok=True)
    params_encoding = f'sl{seed_low}_ha{seed_high_a}_hb{seed_high_b}_cf{cutoff_frac:.2f}'

    infer = sd3i.SD3Inferencer()
    with torch.no_grad():
        infer.load(
            model=model,
            vae=vae,
            shift=3.0,
            controlnet_ckpt=None,
            model_folder=os.path.dirname(model) or ".",
            text_encoder_device=text_encoder_device,
            verbose=verbose,
            load_tokenizers=True,
        )

    latent = infer.get_empty_latent(1, width, height, seed_low, "cpu")
    cond = infer.get_cond(prompt)
    neg_cond = infer.get_cond("")

    # Build the noises
    n_low_source = infer.get_noise(seed_low, latent)
    low_shared, _ = fft_low_high_split(n_low_source, cutoff_frac)
    nA = infer.get_noise(seed_high_a, latent)
    nB = infer.get_noise(seed_high_b, latent)

    noise_A = compose_noise_with_shared_low(low_shared, nA, cutoff_frac)
    noise_B = compose_noise_with_shared_low(low_shared, nB, cutoff_frac)

    # Optional spectral diagnostics
    def amp_spectrum_channel0(x):
        X = torch.fft.fft2(x[0, 0].float(), dim=(-2, -1))
        X = torch.fft.fftshift(X)
        A = torch.log1p(torch.abs(X)).cpu().numpy()
        A = (255.0 * (A - A.min()) / (A.ptp() + 1e-8)).astype(np.uint8)
        return A

    Image.fromarray(amp_spectrum_channel0(n_low_source)).save(out_dir / f"{params_encoding}_amp_low_source.png")
    Image.fromarray(amp_spectrum_channel0(nA)).save(out_dir / f"{params_encoding}_amp_highA_full.png")
    Image.fromarray(amp_spectrum_channel0(nB)).save(out_dir / f"{params_encoding}_amp_highB_full.png")
    Image.fromarray(amp_spectrum_channel0(noise_A)).save(out_dir / f"{params_encoding}_amp_composed_A.png")
    Image.fromarray(amp_spectrum_channel0(noise_B)).save(out_dir / f"{params_encoding}_amp_composed_B.png")
    Image.fromarray(amp_spectrum_channel0(low_shared)).save(out_dir / f"{params_encoding}_amp_low_shared.png")
    
    # Sample
    latA = sample_with_custom_noise(infer, latent, noise_A, cond, neg_cond, steps, cfg, sampler, None, 1.0)
    latB = sample_with_custom_noise(infer, latent, noise_B, cond, neg_cond, steps, cfg, sampler, None, 1.0)

    # Decode and save
    imgA = infer.vae_decode(latA)
    imgB = infer.vae_decode(latB)
    imgA_path = out_dir / f"{params_encoding}_result_A.png"
    imgB_path = out_dir / f"{params_encoding}_result_B.png"
    imgA.save(imgA_path)
    imgB.save(imgB_path)

    # Compare
    A = np.array(imgA.convert("RGB"))
    B = np.array(imgB.convert("RGB"))
    mse_val = mse(A, B)
    psnr_val = psnr(mse_val)

    side = np.concatenate([A, B], axis=1)
    save_image(side, out_dir / f"{params_encoding}_compare_side_by_side.png")

    # plot a grid of 2x2: spectrum of A, spectrum of B, image A, image B
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(2, 2, figsize=(8, 8))
    axs[0, 0].imshow(amp_spectrum_channel0(noise_A), cmap='gray', vmin=0, vmax=255)
    axs[0, 0].set_title('Amplitude Spectrum A')
    axs[0, 0].axis('off')
    axs[0, 1].imshow(amp_spectrum_channel0(noise_B), cmap='gray', vmin=0, vmax=255)
    axs[0, 1].set_title('Amplitude Spectrum B')
    axs[0, 1].axis('off')
    axs[1, 0].imshow(A)
    axs[1, 0].set_title('Image A')
    axs[1, 0].axis('off')
    axs[1, 1].imshow(B)
    axs[1, 1].set_title('Image B')
    axs[1, 1].axis('off')
    plt.tight_layout()
    plt.savefig(out_dir / f"{params_encoding}_summary_plot.png", dpi=300)
    
    report = {
        "prompt": prompt,
        "width": width,
        "height": height,
        "steps": steps,
        "cfg": cfg,
        "sampler": sampler,
        "seed_low": seed_low,
        "seed_high_a": seed_high_a,
        "seed_high_b": seed_high_b,
        "cutoff_frac": cutoff_frac,
        "mse": mse_val,
        "psnr": psnr_val,
        "images": {
            "low_source": str(out_dir / f"{params_encoding}_amp_low_source.png"),
            "low_shared": str(out_dir / f"{params_encoding}_amp_low_shared.png"),
            "highA": str(out_dir / f"{params_encoding}_amp_highA_full.png"),
            "highB": str(out_dir / f"{params_encoding}_amp_highB_full.png"),
            "composed_A": str(out_dir / f"{params_encoding}_amp_composed_A.png"),
            "composed_B": str(out_dir / f"{params_encoding}_amp_composed_B.png"),
            "A": str(imgA_path),
            "B": str(imgB_path),
            "side_by_side": str(out_dir / f"{params_encoding}_compare_side_by_side.png"),
            "summary_plot": str(out_dir / f"{params_encoding}_summary_plot.png"),
        },
    }
    with open(out_dir / f"{params_encoding}_report.json", "w") as f:
        json.dump(report, f, indent=2)

    print(json.dumps(report, indent=2))
    print(f"\nSaved outputs to: {out_dir}")



def main():
    p = argparse.ArgumentParser("Low-frequency hypothesis test for SD3.5 initial noise")
    p.add_argument("--prompt", type=str, default="a studio photo of a ginger cat, soft light")
    p.add_argument("--model", type=str, help="Path to sd3.5 model .safetensors", default="models/sd3.5_medium.safetensors")
    p.add_argument("--clip_g", type=str, help="Path to clip_g.safetensors (if not in folder)", default="models/clip_g.safetensors")
    p.add_argument("--clip_l", type=str, help="Path to clip_l.safetensors (if not in folder)", default="models/clip_l.safetensors")
    p.add_argument("--t5", type=str, help="Path to t5xxl.safetensors (if not in folder)", default="models/t5xxl.safetensors")
    p.add_argument("--vae", type=str, help="Path to sd3_vae.safetensors", default="models/sd3.5_medium.safetensors")
    p.add_argument("--width", type=int, default=1024)
    p.add_argument("--height", type=int, default=1024)
    p.add_argument("--steps", type=int, default=30)
    p.add_argument("--cfg", type=float, default=4.5)
    p.add_argument("--sampler", type=str, default="dpmpp_2m", choices=["dpmpp_2m", "euler"])
    p.add_argument("--seed_low", type=int, default=1234, help="Seed for the SHARED low-frequency component")
    p.add_argument("--seed_high_a", type=int, default=111, help="Seed for A's high-frequencies")
    p.add_argument("--seed_high_b", type=int, default=222, help="Seed for B's high-frequencies")
    p.add_argument("--cutoff_frac", type=float, default=0.10, help="Fraction of Nyquist radius to keep as 'low' (e.g., 0.10)")
    p.add_argument("--out_dir", type=str, default="outputs/lowfreq_test")
    p.add_argument("--text_encoder_device", type=str, default="cpu")
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    # Optionally point encoders via env if you store them alongside model
    model_dir = os.path.dirname(args.model) or "."
    os.environ.setdefault("MODEL_FOLDER", model_dir)
    if args.clip_g: os.environ["CLIP_G_FILE"] = args.clip_g
    if args.clip_l: os.environ["CLIP_L_FILE"] = args.clip_l
    if args.t5:     os.environ["T5XXL_FILE"] = args.t5

    run_experiment(
        prompt=args.prompt,
        model=args.model,
        clip_g=args.clip_g,
        clip_l=args.clip_l,
        t5=args.t5,
        vae=args.vae,
        width=args.width,
        height=args.height,
        steps=args.steps,
        cfg=args.cfg,
        sampler=args.sampler,
        seed_low=args.seed_low,
        seed_high_a=args.seed_high_a,
        seed_high_b=args.seed_high_b,
        cutoff_frac=args.cutoff_frac,
        out_dir=Path(args.out_dir),
        text_encoder_device=args.text_encoder_device,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
