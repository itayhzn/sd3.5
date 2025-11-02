# reinforce_main.py
# Driver for GRPO training with state = latent.flatten() ⊕ sigma_t ⊕ cfg_scale

import argparse
import os
from PIL import Image
import torch

from reinforce import PolicyBank, GRPOTrainer, TrainConfig
from mock_scorer import MockScorer
from sd3_infer import SD3Inferencer

def main(args):
    # with open("pqpp_prompts.txt", "r") as f:
    #     prompts = f.readlines()
    # prompts = [p.strip() for p in prompts if p.strip()]

    if args.prompts is None:
        if args.prompts_file is not None:
            with open(args.prompts_file, "r") as f:
                args.prompts = [line.strip() for line in f.readlines() if line.strip()]
        else:
            args.prompts = ["a studio photo of a ginger cat, soft light"]
    
    out_dir = os.path.join(args.out_dir, args.experiment_name)

    conf = TrainConfig(
        schedule=args.schedule,
        group_size=args.group_size,
        num_epochs=args.num_epochs,
        lr=args.lr,
        max_grad_norm=args.max_grad_norm,
        save_every=args.save_every,
        out_dir=out_dir,
        resume_from=args.resume_from,
        prompts=args.prompts,
        seeds=args.seeds,
        width=args.width,
        height=args.height,
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
        mode=args.action_mode,     # try 'basis_delta' first; 'latent_delta' also supported
        action_dim_basis=args.action_dim_basis,
        action_alpha=args.action_alpha,
        state_alpha=args.state_alpha,
        out_dir=out_dir,
    )

    trainer = GRPOTrainer(
        inferencer=inf,
        steps=args.steps,
        cfg_scale=args.cfg_scale,
        sampler="dpmpp_2m",
        out_dir=out_dir,
    )

    mock = MockScorer(mode=args.reward_scorer)
    def reward_fn(prompt: str, img: Image.Image) -> float:
        return mock(img)


    os.makedirs(conf.out_dir, exist_ok=True)
    trainer.train(bank, reward_fn, conf)
    print("Done. Check:", conf.out_dir)

if __name__ == "__main__":
    # use argparse to get parameters from command line if needed
    

    parser = argparse.ArgumentParser()
    parser.add_argument("--schedule", type=tuple, default=(0,))
    parser.add_argument("--group_size", type=int, default=8)
    parser.add_argument("--num_epochs", type=int, default=15)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--save_every", type=int, default=1)
    parser.add_argument("--out_dir", type=str, default="outputs/grpo")
    parser.add_argument("--resume_from", type=str, default=None)
    parser.add_argument("--prompts", type=list, default=[])
    parser.add_argument("--prompts_file", type=str, default=None)
    parser.add_argument("--seeds", type=list, default=[])
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--action_alpha", type=float, default=1.0)
    parser.add_argument("--state_alpha", type=float, default=0.02)
    parser.add_argument("--action_mode", type=str, default="basis_delta", choices=["basis_delta", "latent_delta"])  
    parser.add_argument("--hidden_size", type=int, default=256)
    parser.add_argument("--action_dim_basis", type=int, default=64)
    parser.add_argument("--experiment_name", type=str, default="default")
    parser.add_argument("--reward_scorer", type=str, default="brightness", choices=["sharp_contrast", "brightness", "entropy"])
    parser.add_argument("--steps", type=int, default=28)
    parser.add_argument("--cfg_scale", type=float, default=4.5)
    args = parser.parse_args()
    
    main(args)