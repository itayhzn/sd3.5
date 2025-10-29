# reinforce_main.py
# Driver for GRPO training with state = latent.flatten() ⊕ sigma_t ⊕ cfg_scale

import os
from PIL import Image
import torch

from reinforce import PolicyBank, GRPOTrainer, TrainConfig
from mock_scorer import MockScorer
from sd3_infer import SD3Inferencer

def main():
    with open("pqpp_prompts.txt", "r") as f:
        prompts = f.readlines()
    prompts = [p.strip() for p in prompts if p.strip()]

    conf = TrainConfig(
        schedule=(0,),
        group_size=4,
        num_epochs=5,
        lr=0.01,
        max_grad_norm=1.0,
        save_every=1,
        out_dir="outputs/grpo_mock_scorer",
        resume_from=None,
        prompts=prompts[:100],
        seeds=[23,34,45,56],
        width=1024,
        height=1024,
    )

    # Frozen SD3.5
    inf = SD3Inferencer()
    with torch.no_grad():
        inf.load(
            model="models/sd3.5_medium.safetensors",
            vae=None,
            shift=3.0,
            controlnet_ckpt=None,
            model_folder="models",
            text_encoder_device="cpu",
            verbose=False,
        )

    bank = PolicyBank(
        mode="basis_delta",     # try 'basis_delta' first; 'latent_delta' also supported
        action_dim_basis=64,
        alpha=0.05,
        device="cuda",
    )

    trainer = GRPOTrainer(
        inferencer=inf,
        steps=28,
        cfg_scale=4.5,
        sampler="dpmpp_2m",
        device="cuda",
    )

    mock = MockScorer(mode="sharp_contrast")
    def reward_fn(prompt: str, img: Image.Image) -> float:
        return mock(img)

    os.makedirs(conf.out_dir, exist_ok=True)
    trainer.train(bank, reward_fn, conf)
    print("Done. Check:", conf.out_dir)

if __name__ == "__main__":
    main()