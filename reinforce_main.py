# test_reinforce_mock.py
import os
from PIL import Image

from reinforce import PolicyBank, SingleStepTrainer, TrainConfig
from mock_scorer import MockScorer
from sd3_infer import SD3Inferencer  # adjust import to your file
import torch

# ---------- Config ----------
# Note: Sigma shift value, publicly released models use 3.0
SHIFT = 3.0
# Naturally, adjust to the width/height of the model you have
WIDTH = 1024
HEIGHT = 1024
# Pick your prompt
PROMPT = "a photo of a cat"
# Most models prefer the range of 4-5, but still work well around 7
CFG_SCALE = 4.5
# Different models want different step counts but most will be good at 50, albeit that's slow to run
# sd3_medium is quite decent at 28 steps
STEPS = 40
# Seed
SEED = 23
SEEDTYPE = "fixed"
# SEEDTYPE = "rand"
# SEEDTYPE = "roll"
# Actual model file path
MODEL = "models/sd3.5_medium.safetensors"
# MODEL = "models/sd3.5_large_turbo.safetensors"
# MODEL = "models/sd3.5_large.safetensors"
# VAE model file path, or set None to use the same model file
VAEFile = None  # "models/sd3_vae.safetensors"
# Optional init image file path
INIT_IMAGE = None
# ControlNet
CONTROLNET_COND_IMAGE = None
# If init_image is given, this is the percentage of denoising steps to run (1.0 = full denoise, 0.0 = no denoise at all)
DENOISE = 0.8
# Output file path
OUTDIR = "outputs"
# SAMPLER
SAMPLER = "dpmpp_2m"
# MODEL FOLDER
MODEL_FOLDER = "models"

SCHEDULE = (2, )                                 # intervene at one step first
EPOCHS = 2

if __name__ == "__main__":
    # ---------- Build inferencer ----------
    inf = SD3Inferencer()
    with torch.no_grad():
        inf.load(
            model=MODEL,
            vae=None,
            shift=SHIFT,
            controlnet_ckpt=None,
            model_folder=MODEL_FOLDER,
            text_encoder_device="cpu",
            verbose=False,
        )

    # ---------- RL objects ----------
    bank = PolicyBank(
        mode="basis_delta",       # or "latent_delta" (try basis first; more stable)
        action_dim_basis=32,      # small k for speed
        alpha=0.02,               # small nudge
        device="cuda",
    )

    trainer = SingleStepTrainer(
        inferencer=inf,
        prompt=PROMPT,
        width=WIDTH, height=HEIGHT,
        steps=STEPS,
        cfg=CFG_SCALE,
        sampler=SAMPLER,
        seed=SEED,
        device="cuda",
    )

    # Mock reward
    reward_fn = MockScorer(mode="sharp_contrast")

    cfg = TrainConfig(
        schedule=SCHEDULE,
        num_epochs=EPOCHS,
        iters_per_t=2,            # two updates per t per epoch
        lr=3e-4,
        value_coef=0.5,
        max_grad_norm=1.0,
        save_every=1,
        out_dir=OUTDIR,
    )

    # ---------- Run training ----------
    trainer.train(bank, reward_fn, cfg)

    print("Done. Check:", OUTDIR)