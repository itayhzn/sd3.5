# reinforce.py
# Per-timestep, single-intervention REINFORCE for SD3.5
# Batch over prompts + seeds; supports both 'latent_delta' and 'basis_delta'.

from dataclasses import dataclass
from typing import Callable, Dict, Optional, Sequence, Tuple, List, Mapping
import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import sd3_impls as _sd3  # must expose CFGDenoiser(model)

# ------------------ utils ------------------

def logger(d: str, csv_file: str):
    keys = sorted(d.keys())
    msg = ",".join([str(d[k]) for k in keys])
    base_dir = os.path.dirname(csv_file)
    os.makedirs(base_dir, exist_ok=True)
    
    if not os.path.exists(csv_file):
        with open(csv_file, "w") as f:
            f.write(",".join(keys) + "\n")
            f.write(msg + "\n")
    else:
        with open(csv_file, "a") as f:
            f.write(msg + "\n")

def normalize(t: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return t / (t.norm(p=2) + eps)

def gaussian_logprob(a: torch.Tensor, mu: torch.Tensor, log_std: torch.Tensor) -> torch.Tensor:
    var = (log_std.exp())**2
    return -0.5 * ( ((a - mu)**2) / var + 2*log_std + math.log(2*math.pi) ).sum(-1)

def orthonormal_basis(dim: int, k: int, device: str = "cuda", dtype=torch.float32) -> torch.Tensor:
    A = torch.randn(dim, k, device=device, dtype=dtype)
    Q, _ = torch.linalg.qr(A, mode="reduced")
    return Q  # (dim, k)

def sinusoidal_embed(x: float, dim: int = 16, device: str = "cuda") -> torch.Tensor:
    t = torch.tensor([x], device=device)
    half = dim // 2
    freqs = torch.exp(torch.arange(0, half, device=device, dtype=torch.float32) * (-math.log(10_000.0) / max(1, half-1)))
    ang = t[:, None] * freqs[None, :]
    emb = torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1).squeeze(0)
    if emb.numel() < dim: emb = F.pad(emb, (0, dim - emb.numel()))
    return emb

# ------------------ tiny nets ------------------

class TinyMLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, out_dim),
        )
    def forward(self, x): return self.net(x)

# ------------------ state builder ------------------

class StateBuilder:
    """
    Per-step state s_t from latent + time:
      - latent stats (6): mean, std, |.|_mean, L2/√N, min, max
      - t_frac embedding (16)
      - cfg_scale, t_frac (2)
    -> padded/truncated to state_dim.
    """
    def __init__(self, state_dim: int = 128, t_emb_dim: int = 16, device: str = "cuda"):
        self.state_dim = state_dim
        self.t_emb_dim = t_emb_dim
        self.device = device

    def build(self, latent: torch.Tensor, t_idx: int, T: int, cfg_scale: float) -> torch.Tensor:
        z = latent
        with torch.no_grad():
            s = latent.flatten() 
        return s  # (state_dim,)

# ------------------ per-step policy & critic ------------------

class StepPolicy(nn.Module):
    """
    Tiny Gaussian policy head for ONE timestep t.
    Modes:
      - 'latent_delta' : action a ∈ R^D (D=flattened latent)
      - 'basis_delta'  : action u ∈ R^k, delta z = B @ u
    """
    def __init__(
        self,
        mode: str,
        state_dim: int,
        action_dim: int,          # k for basis; placeholder for latent (resized later)
        init_log_std: float = -1.0,
        alpha: float = 0.02,
        device: str = "cuda",
    ):
        super().__init__()
        assert mode in ("latent_delta", "basis_delta")
        self.mode = mode
        self.alpha = alpha
        self.device = device

        self.actor = TinyMLP(state_dim, 2*action_dim).to(device)  # last layer resized for latent_delta
        with torch.no_grad():
            last = [m for m in self.actor.modules() if isinstance(m, nn.Linear)][-1]
            A = last.out_features // 2
            last.bias[A:] = init_log_std

        self.critic = TinyMLP(state_dim, 1).to(device)

        self.latent_dim: Optional[int] = None
        self.basis: Optional[torch.Tensor] = None

    def ensure_shapes(self, latent: torch.Tensor, action_dim_basis: int):
        D = latent.numel()
        if self.mode == "latent_delta":
            if self.latent_dim is None or (2*self.latent_dim) != self.actor.net[-1].out_features:
                self.latent_dim = D
                head_in = self.actor.net[-1].in_features
                new_last = nn.Linear(head_in, 2*D).to(self.device)
                with torch.no_grad():
                    new_last.bias[D:] = -1.0
                self.actor.net[-1] = new_last
        else:
            if self.basis is None:
                self.basis = orthonormal_basis(D, action_dim_basis, device=self.device, dtype=torch.float32)

    def act(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        params = self.actor(state)                 # (2*A,)
        A = params.numel() // 2
        mu, log_std = params[:A], params[A:]
        std = log_std.exp()
        eps = torch.randn_like(std)
        a = mu + std * eps
        logp = gaussian_logprob(a, mu, log_std)    # scalar
        return a, logp

    def value(self, state: torch.Tensor) -> torch.Tensor:
        return self.critic(state).squeeze(-1)

# ------------------ policy bank ------------------

class PolicyBank(nn.Module):
    """
    Independent StepPolicy (+critic) for each t.
    Exposes `get_policy(latent_t, t, T, cfg)` and `apply_action(...)`.
    """
    def __init__(
        self,
        mode: str = "basis_delta",
        state_dim: int = 128,
        action_dim_basis: int = 64,
        alpha: float = 0.02,
        device: str = "cuda",
    ):
        super().__init__()
        self.mode = mode
        self.state_dim = state_dim
        self.action_dim_basis = action_dim_basis
        self.alpha = alpha
        self.device = device
        self.bank: nn.ModuleDict = nn.ModuleDict()  # store as modules for .parameters()
        self.state_builder = StateBuilder(state_dim=state_dim, device=device)

    def policy(self, t: int) -> StepPolicy:
        key = str(int(t))
        if key not in self.bank:
            self.bank[key] = StepPolicy(
                mode=self.mode,
                state_dim=self.state_dim,
                action_dim=(self.action_dim_basis if self.mode == "basis_delta" else 2),
                alpha=self.alpha,
                device=self.device,
            )
        return self.bank[key]

    def get_policy(self, latent_t: torch.Tensor, t: int, T: int, cfg_scale: float):
        pol = self.policy(t)
        pol.ensure_shapes(latent_t, action_dim_basis=self.action_dim_basis)
        s_t = self.state_builder.build(latent_t, t, T, cfg_scale).to(self.device)
        a_t, logp_t = pol.act(s_t)
        v_t = pol.value(s_t)
        logger({"step": t, "a_t": a_t.norm().item(), "mean(a)": a_t.mean().item(), "std(a)": a_t.std().item(), "v_t": v_t.item(), "logp_t": logp_t.item()}, 'logs/get_policy_logs.csv')
        return a_t, logp_t, v_t, s_t

    def apply_action(self, latent: torch.Tensor, a: torch.Tensor, t: int) -> torch.Tensor:
        pol = self.policy(t)
        if pol.mode == "latent_delta":
            delta = a.reshape_as(latent)
        else:
            flat = pol.basis @ a
            delta = flat.view_as(latent)
        delta = normalize(delta) * pol.alpha
        return latent + delta

    def reset_policies(self, latent: torch.Tensor, schedule: Sequence[int]):
        for t in schedule:
            pol = self.policy(t)
            pol.ensure_shapes(latent, action_dim_basis=self.action_dim_basis)

# ------------------ denoiser wrapper (single step) ------------------

class SingleStepDenoiserWrapper:
    """
    Wraps CFGDenoiser so ONLY step t* applies a policy nudge to x.
    Mirrors CFGDenoiser.forward signature.
    """
    def __init__(self, base_denoiser, bank: PolicyBank, target_t: int, cfg_scale: float, total_steps: int):
        self.base = base_denoiser
        self.bank = bank
        self.t_star = int(target_t)
        self.cfg_scale = float(cfg_scale)
        self.T = int(total_steps)

        # logging for RL update
        self.logged = False
        self.logged_state = None
        self.logged_logp = None
        self.logged_value = None
        self.logged_action = None

        self.t_idx = 0  # step counter

    def forward(self, x, timestep, cond, uncond, cond_scale, save_tensors_path=None, **kwargs):
        if self.t_idx == self.t_star and not self.logged:
            a_t, logp_t, v_t, s_t = self.bank.get_policy(x, self.t_idx, self.T, self.cfg_scale)
            x = self.bank.apply_action(x, a_t, self.t_idx)
            self.logged = True
            self.logged_state = s_t
            self.logged_logp = logp_t
            self.logged_value = v_t
            self.logged_action = a_t

        out = self.base.forward(
            x, timestep, cond, uncond, cond_scale,
            save_tensors_path=save_tensors_path, **kwargs
        )
        self.t_idx += 1
        return out

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

# ------------------ trainer ------------------

@dataclass
class TrainConfig:
    # RL & logging
    schedule: Tuple[int, ...] = (10, 20, 35)
    num_epochs: int = 3
    iters_per_t: int = 1
    lr: float = 3e-4
    value_coef: float = 0.5
    entropy_coef: float = 0.0
    max_grad_norm: float = 1.0
    save_every: int = 1
    
    out_dir: str = "outputs/rl_per_timestep"
    resume_from: Optional[str] = None

    # Batch prompts/seeds, resolution
    prompts: Sequence[str] = ("a photo of a cat",)
    seeds: Sequence[int] = (23,)
    width: int = 1024
    height: int = 1024

    # model config
    cfg_scale: float = 4.5
    num_diffusion_steps: int = 40
    shift: float = 3.0
    model: str = "models/sd3.5_medium.safetensors"
    model_folder: str = "models"
    sampler: str = "dpmpp_2m"

class SingleStepTrainer:
    """
    Baseline per epoch: compute baseline per prompt (no intervention).
    For each t in union(schedule): run one rollout per prompt with an intervention at t,
    accumulate losses across prompts, then do one optimizer step.
    """
    def __init__(self, inferencer, steps: int, cfg: float, sampler: str, device: str = "cuda"):
        self.inf = inferencer
        self.steps = steps
        self.cfg = cfg
        self.sampler = sampler
        self.device = device

        # cache negative conditioning once
        self.neg_cond = self.inf.get_cond("")

    @torch.no_grad()
    def _run_once_prompt(self, prompt: str, seed: int, width: int, height: int, wrapper=None, tag=None, save_dir=None) -> Image.Image:
        latent = self.inf.get_empty_latent(1, width, height, seed, device="cuda")
        cond = self.inf.get_cond(prompt)      # tuple; do_sampling will fix_cond()
        ncond = self.neg_cond

        original_cfg = _sd3.CFGDenoiser
        if wrapper is None:
            pass
        else:
            def _fake_cfg(*args, **kwargs):
                return wrapper
            _sd3.CFGDenoiser = _fake_cfg

        try:
            sampled_latent = self.inf.do_sampling(
                latent=latent,
                seed=seed,
                conditioning=cond,
                neg_cond=ncond,
                steps=self.steps,
                cfg_scale=self.cfg,
                sampler=self.sampler,
                controlnet_cond=None,
                denoise=1.0,
                skip_layer_config={},
                save_tensors_path=None,
            )
        finally:
            _sd3.CFGDenoiser = original_cfg

        img = self.inf.vae_decode(sampled_latent)
        if save_dir and tag:
            os.makedirs(save_dir, exist_ok=True)
            img.save(os.path.join(save_dir, f"{tag}.png"))
        return img

    def train(
        self,
        bank: PolicyBank,
        reward_fn: Callable[[str, Image.Image], float],   # (prompt, image) -> float
        conf: TrainConfig
    ):
        prompts: List[str] = list(conf.prompts)
        seeds:   List[int] = list(conf.seeds)
        
        P = len(prompts)
        S = len(seeds)

        # Materialize policies for union of steps, size against a latent of the target resolution
        union_steps = sorted({int(t) for t in conf.schedule})
        with torch.no_grad():
            # create a dummy latent of the correct shape for sizing/basis
            z0 = self.inf.get_empty_latent(1, conf.width, conf.height, seeds[0], device="cuda")
        bank.reset_policies(z0, union_steps)

        # Optimizer over all policies/critics
        opt = torch.optim.AdamW(bank.parameters(), lr=conf.lr, betas=(0.9, 0.999), weight_decay=1e-4)

        # Optionally resume (weights only)
        if conf.resume_from is not None and os.path.exists(conf.resume_from):
            state = torch.load(conf.resume_from, map_location="cpu")
            bank.load_state_dict(state["policy_bank"])
            print(f"Resumed policy_bank from {conf.resume_from}")

        for epoch in range(1, conf.num_epochs + 1):
            # 1) Baselines per prompt
            R_base: List[float] = [0.0] * (P * S)
            for i, pr in enumerate(prompts):
                for j, sd in enumerate(seeds):
                    idx = i * S + j
                    save_dir = (conf.out_dir if (epoch % conf.save_every) == 0 else None)
                    tag = f"epoch{epoch:02d}_base_p{i:02d}_s{j:02d}"
                    img_base = self._run_once_prompt(prompt=pr, seed=sd, width=conf.width, height=conf.height,
                                                    wrapper=None, tag=tag, save_dir=save_dir)
                    R_base[idx] = float(reward_fn(pr, img_base))
                    
            # 2) Train each step on the batch of prompts
            for t in union_steps:
                opt.zero_grad(set_to_none=True)
                total_policy_loss = 0.0
                total_value_loss  = 0.0
                n_contrib = 0

                for i, pr in enumerate(prompts):
                    for j, sd in enumerate(seeds):
                        # build base denoiser + wrapper that intervenes at this t
                        base = _sd3.CFGDenoiser(self.inf.sd3.model)
                        wrapper = SingleStepDenoiserWrapper(
                            base_denoiser=base,
                            bank=bank,
                            target_t=t,
                            cfg_scale=self.cfg,
                            total_steps=self.steps,
                        )

                        # rollout 1x with intervention at t
                        save_dir = (conf.out_dir if (epoch % conf.save_every) == 0 else None)
                        tag = f"epoch{epoch:02d}_t{t:03d}_p{i:02d}_s{j:02d}"
                        img_t = self._run_once_prompt(prompt=pr, seed=sd, width=conf.width, height=conf.height,
                                                    wrapper=wrapper, tag=tag, save_dir=save_dir)
                        R_t = float(reward_fn(pr, img_t))

                        # If wrapper didn't fire (shouldn't), skip
                        if wrapper.logged_state is None:
                            continue

                        V_t = wrapper.logged_value
                        logp_t = wrapper.logged_logp

                        # Advantage with strong baseline per prompt
                        idx = i * S + j
                        A_hat = torch.tensor([R_t - R_base[idx]], device=bank.device, dtype=torch.float32) - V_t.detach()
                        policy_loss = -(A_hat * logp_t).mean()
                        value_loss  = conf.value_coef * ((R_t - R_base[idx]) - V_t).pow(2).mean()

                        total_policy_loss = total_policy_loss + policy_loss
                        total_value_loss  = total_value_loss  + value_loss
                        n_contrib += 1

                        logger({"epoch": epoch, "t": t, "prompt_idx": i, "seed_idx": j, "R_t": R_t, "R_base": R_base[idx], "A_hat": A_hat.item(), "policy_loss": policy_loss.item(), "value_loss": value_loss.item()}, 'logs/optimization_logs.csv')

                if n_contrib > 0:
                    logger({"epoch": epoch, "t": t, "total_policy_loss": total_policy_loss.item(), "total_value_loss": total_value_loss.item(), "n_contrib": n_contrib}, 'logs/epoch_step_logs.csv')
                    loss = (total_policy_loss + total_value_loss) / n_contrib
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(bank.parameters(), conf.max_grad_norm)
                    opt.step()

            # Save checkpoint each save_every
            # if epoch % conf.save_every == 0:
            #     os.makedirs(conf.out_dir, exist_ok=True)
            #     ckpt_path = os.path.join(conf.out_dir, f"policy_bank_epoch_{epoch:02d}.pt")
            #     torch.save({"policy_bank": bank.state_dict(), "schedule": conf.schedule, "epoch": epoch}, ckpt_path)
            #     print(f"Saved checkpoint to {ckpt_path}")

        print("Training complete.")