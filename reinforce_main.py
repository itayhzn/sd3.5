# reinforce_main.py
# Batch training driver with a prompt-aware reward shim.

import os
from PIL import Image
import torch

from reinforce import PolicyBank, SingleStepTrainer, TrainConfig
from mock_scorer import MockScorer
from sd3_infer import SD3Inferencer

def main():
    # ---------- Train config ----------
    conf = TrainConfig(
        schedule=(0),
        num_epochs=100,
        iters_per_t=1,
        lr=0.01,
        value_coef=0.5,
        max_grad_norm=1.0,
        save_every=1,
        out_dir='outputs/rl_batch',
        resume_from=None,

        prompts=[
            "a studio photo of a ginger cat, soft light"
        ],
        seeds=[23],
        width=1024,
        height=1024,

        # model config
        cfg_scale=4.5,
        num_diffusion_steps=28,
        shift=3.0,
        model="models/sd3.5_medium.safetensors",
        model_folder="models",
        sampler="dpmpp_2m",

    )

    # ---------- Build inferencer ----------
    inf = SD3Inferencer()
    with torch.no_grad():
        inf.load(
            model=conf.model,
            vae=None,
            shift=conf.shift,
            controlnet_ckpt=None,
            model_folder=conf.model_folder,
            text_encoder_device="cpu",
            verbose=False,
        )

    # ---------- RL objects ----------
    bank = PolicyBank(
        mode="basis_delta",   # 'basis_delta' is usually more stable than 'latent_delta'
        action_dim_basis=64,
        alpha=0.02,
        device="cuda",
    )

    trainer = SingleStepTrainer(
        inferencer=inf,
        steps=conf.num_diffusion_steps,
        cfg=conf.cfg_scale,
        sampler=conf.sampler,
        device="cuda",
    )

    # Prompt-aware reward shim (keeps MockScorer compatible now, easy swap to ImageReward later)
    mock = MockScorer(mode="brightness")
    def reward_fn(prompt: str, img: Image.Image) -> float:
        return mock(img)

    os.makedirs(conf.out_dir, exist_ok=True)
    trainer.train(bank, reward_fn, conf)
    print("Done. Check:", conf.out_dir)

if __name__ == "__main__":
    main()