# reinforce.py
# GRPO over per-timestep latent nudges for SD3.5
# State = concat(latent.flatten(), [sigma_t], [cfg_scale])

from dataclasses import dataclass
from typing import Callable, Optional, Sequence, Tuple, List

import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from PIL import Image
import sys

import sd3_infer as _sdi
import sd3_impls as _sdm

original_cfg_infer = getattr(_sdi, "CFGDenoiser", None)
original_cfg_impl  = getattr(_sdm, "CFGDenoiser", None)

# ------------------ utils ------------------

def logger(d: str, csv_file: str):
    with open(csv_file, "a") as f:
        f.write(str(d) + "\n")

def normalize(t: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return t / (t.norm(p=2) + eps)

def orthonormal_basis(dim: int, k: int, device: str = "cuda", dtype=torch.float32) -> torch.Tensor:
    A = torch.randn(dim, k, device=device, dtype=dtype)
    Q, _ = torch.linalg.qr(A, mode="reduced")
    return Q  # (dim, k)

def make_mlp(in_dim: int, out_dim: int, hidden: int = 256) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, hidden), nn.SiLU(),
        nn.Linear(hidden, hidden), nn.SiLU(),
        nn.Linear(hidden, out_dim),
    )


# ------------------ state builder ------------------

class StateBuilder:
    """
    s_t = concat( latent.flatten(), [sigma_t], [cfg_scale] )
    """
    def __init__(self, device: str = "cuda"):
        self.device = device


    @torch.no_grad()
    def build(self, latent: torch.Tensor, sigma_t: float, cfg_scale: float) -> torch.Tensor:
        # latent: (B=1, C=16, H, W) in SD3.5
        x = latent.detach().to(self.device, dtype=torch.float32)
        if x.dim() == 4 and x.size(0) == 1:
            x = x.squeeze(0)            # (C, H, W)
        if x.dim() == 3:
            # reduce channels with mean; (you can try std or [mean,std] concat later)
            x = x.mean(dim=0)           # (H, W)
        elif x.dim() == 2:
            pass                        # already (H, W)
        else:
            raise ValueError(f"Unexpected latent shape for state: {tuple(latent.shape)}")

        H, W = x.shape
        # target ~64 elements, keep aspect, and do not upsample
        target_elems = 64
        scale = min(1.0, math.sqrt(target_elems / max(1, H * W)))
        h_new = max(1, int(H * scale))
        w_new = max(1, int(W * scale))

        # downsample
        x = F.adaptive_avg_pool2d(x.unsqueeze(0).unsqueeze(0), output_size=(h_new, w_new)) \
                .squeeze(0).squeeze(0)   # (h_new, w_new)

        flat = x.flatten()               # ~64 dims

        # add log-sigma and cfg
        sigma_feat = torch.tensor([math.log(max(1e-8, float(sigma_t)))],
                                  device=self.device, dtype=torch.float32)
        cfg_feat   = torch.tensor([float(cfg_scale)], device=self.device, dtype=torch.float32)

        s = torch.cat([flat, sigma_feat, cfg_feat], dim=0)  # (≈66,)

        # optional light normalization (helps training stability)
        if s.numel() > 2:
            s_center = s[:-2]
            s_std = s_center.std(unbiased=False).clamp_min(1e-6)
            s_center = (s_center - s_center.mean()) / s_std
            s = torch.cat([s_center, s[-2:]], dim=0)

        return s


# ------------------ per-step policy ------------------

class StepPolicy(nn.Module):
    """
    Tiny Gaussian policy head for ONE timestep t.
    Modes:
      - 'latent_delta' : action a ∈ R^D (D=flattened latent)
      - 'basis_delta'  : action u ∈ R^k, delta z = B @ u
    Actor adapts its input (D+2) and, for latent_delta, its output (2*D) when latent size changes.
    """
    def __init__(
        self,
        mode: str,
        action_dim_basis: int,
        init_log_std: float = -1.0,
        alpha: float = 0.02,
        hidden: int = 256,
        device: str = "cuda",
    ):
        super().__init__()
        assert mode in ("latent_delta", "basis_delta")
        self.mode = mode
        self.alpha = alpha
        self.hidden = hidden
        self.device = device

        # lazily built after we see a latent (to know D)
        self.actor: nn.Sequential = None  # type: ignore
        self.actor_in_dim: Optional[int] = None
        self.action_dim_basis = action_dim_basis
        self.init_log_std = init_log_std

        # for basis mode + size tracking
        self.basis: Optional[torch.Tensor] = None
        self.latent_dim: Optional[int] = None

    def _rebuild_actor(self, state_dim: int, D: int):
        if self.mode == "latent_delta":
            out_dim = 2 * D
        else:
            out_dim = 2 * self.action_dim_basis
        self.actor = make_mlp(state_dim, out_dim, hidden=self.hidden).to(self.device)
        with torch.no_grad():
            last = [m for m in self.actor.modules() if isinstance(m, nn.Linear)][-1]
            half = last.out_features // 2
            last.bias[half:] = self.init_log_std
        self.actor_in_dim = state_dim

    def ensure_shapes(self, latent: torch.Tensor, state_dim: int, action_dim_basis: int):
        D = latent.numel()
        # (re)build actor if state_dim or (for latent_delta) D changed
        if (self.actor is None) or (self.actor_in_dim != state_dim) \
           or (self.mode == "latent_delta" and self.latent_dim != D):
            if self.mode == "latent_delta":
                out_dim = 2 * D
            else:
                out_dim = 2 * self.action_dim_basis
            self.actor = make_mlp(state_dim, out_dim, hidden=self.hidden).to(self.device)
            with torch.no_grad():
                last = [m for m in self.actor.modules() if isinstance(m, nn.Linear)][-1]
                half = last.out_features // 2
                last.bias[half:] = self.init_log_std
            self.actor_in_dim = state_dim
            self.latent_dim = D
        if self.mode == "basis_delta" and (self.basis is None or self.basis.shape[0] != D):
            self.basis = orthonormal_basis(D, self.action_dim_basis, device=self.device, dtype=torch.float32)

    # StepPolicy
    def sample(self, state: torch.Tensor, generator: Optional[torch.Generator] = None):
        params = self.actor(state)  # (2*A,)
        A = params.numel() // 2
        mu, log_std = params[:A], params[A:]
        std = log_std.exp().clamp(min=1e-5)
        
        if generator is None:
            eps = torch.randn_like(std)
        else:
            eps = torch.randn(std.shape, device=std.device, dtype=std.dtype, generator=generator)

        a = mu + std * eps
        # log prob (sum over dims)
        logp = (-0.5 * (((a - mu) / std) ** 2 + 2 * log_std + math.log(2 * math.pi))).sum(-1)
        return a.detach(), logp  # detach action (we only backprop through logp)


# ------------------ policy bank ------------------

class PolicyBank(nn.Module):
    """
    Independent StepPolicy for each t. Also holds StateBuilder.
    """
    def __init__(
        self,
        mode: str = "basis_delta",
        action_dim_basis: int = 64,
        alpha: float = 0.02,
        hidden: int = 256,
        device: str = "cuda",
    ):
        super().__init__()
        self.mode = mode
        self.action_dim_basis = action_dim_basis
        self.alpha = alpha
        self.hidden = hidden
        self.device = device
        self.bank = nn.ModuleDict()  # keyed by str(t)
        self.state_builder = StateBuilder(device=device)
        
    def policy(self, t: int) -> StepPolicy:
        key = str(int(t))
        if key not in self.bank:
            self.bank[key] = StepPolicy(
                mode=self.mode,
                action_dim_basis=self.action_dim_basis,
                alpha=self.alpha,
                hidden=self.hidden,
                device=self.device,
            )
        return self.bank[key]

    def get_state(self, latent_t: torch.Tensor, t: int, sigma_t: float, cfg_scale: float) -> torch.Tensor:
        s_t = self.state_builder.build(latent_t, sigma_t, cfg_scale).to(self.device)
        pol = self.policy(t)
        pol.ensure_shapes(latent_t, state_dim=s_t.numel(), action_dim_basis=self.action_dim_basis)
        return s_t

    def apply_action(self, latent: torch.Tensor, a: torch.Tensor, t: int) -> torch.Tensor:
        pol = self.policy(t)
        if pol.mode == "latent_delta":
            delta = a.reshape_as(latent)
        else:
            flat = pol.basis @ a
            delta = flat.view_as(latent)
        delta = normalize(delta) * pol.alpha
        print(latent.norm().item(), delta.norm().item(), (latent + delta).norm().item())
        return latent + delta

    def reset_policies(self, latent: torch.Tensor, schedule: Sequence[int]):
        # force construction/sizing for initial latent shape
        s_t = self.state_builder.build(latent, sigma_t=1.0, cfg_scale=1.0).to(self.device)
        for t in schedule:
            self.policy(t).ensure_shapes(latent, state_dim=s_t.numel(), action_dim_basis=self.action_dim_basis)


# ------------------ denoiser wrapper (single step, GRPO) ------------------

class GRPODenoiserWrapper:
    """
    At target step t*, build state s_t (latent.flatten() ⊕ sigma_t ⊕ cfg_scale),
    sample action, apply nudge, and log log-probability.
    """
    def __init__(
        self,
        base_denoiser,
        schedule: Sequence[int],
        bank: PolicyBank,
        cfg_scale: float,
        sigmas: torch.Tensor,     # 1D tensor of sigmas used by sampler (no trailing 0)
        total_steps: int,
        generator: Optional[torch.Generator] = None,
    ):
        self.base = base_denoiser
        self.bank = bank
        self.schedule = set(schedule)
        self.cfg_scale = float(cfg_scale)
        self.sigmas = sigmas.detach().float().cpu()  # shape: [steps] (no last 0)
        self.T = int(total_steps)

        self.logged_logp = None
        self.t_idx = 0
        self._acted = False

        self.generator = generator or torch.Generator(device=bank.device)
        if not self.generator.initial_seed():
            self.generator.manual_seed(torch.seed())


    def forward(self, x, timestep, cond, uncond, cond_scale, save_tensors_path=None, **kwargs):
        if (not self._acted) and (self.t_idx in self.schedule):
            sigma_t = float(self.sigmas[min(self.t_idx, len(self.sigmas) - 1)])
            x_detach = x.detach()
            with torch.enable_grad():
                s_t = self.bank.get_state(x_detach, self.t_idx, sigma_t, self.cfg_scale)
                a_t, logp_t = self.bank.policy(self.t_idx).sample(s_t, generator=self.generator)

            x = self.bank.apply_action(x_detach, a_t.detach(), self.t_idx)
            self.logged_logp = logp_t
            self._acted = True
            # logger({
            #     "t_idx": self.t_idx,
            #     "sigma_t": sigma_t,
            #     "cfg_scale": self.cfg_scale,
            #     "action_norm": a_t.norm().item(),
            #     "logp": logp_t.item(),
            # }, f"outputs/grpo_mock_scorer/policy_log_{self.t_idx:03d}.log")
            # # save latent and action for debugging
        
            # torch.save(x.cpu(), f"outputs/grpo_mock_scorer/latent_t{self.t_idx:03d}.pt")
            # torch.save(a_t.cpu(), f"outputs/grpo_mock_scorer/action_t{self.t_idx:03d}.pt")

        out = self.base.forward(
            x, timestep, cond, uncond, cond_scale,
            save_tensors_path=save_tensors_path, **kwargs
        )
        self.t_idx += 1
        return out

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


# ------------------ trainer (GRPO) ------------------

@dataclass
class TrainConfig:
    # GRPO
    schedule: Tuple[int, ...] = (10, 20, 35)
    group_size: int = 4
    num_epochs: int = 2
    lr: float = 3e-4
    max_grad_norm: float = 1.0
    save_every: int = 1
    out_dir: str = "outputs/grpo_per_timestep"
    resume_from: Optional[str] = None

    # Batch prompts/seeds, resolution
    prompts: Sequence[str] = ("a photo of a cat",)
    seeds: Sequence[int] = (23,)
    width: int = 1024
    height: int = 1024


class GRPOTrainer:
    """
    For each (prompt, seed, t): sample G actions (group), compute rewards, normalise advantages,
    loss = -mean(A_i * logpi_i). No critic.
    """
    def __init__(self, inferencer, steps: int, cfg_scale: float, sampler: str, device: str = "cuda"):
        self.inf = inferencer
        self.steps = steps
        self.cfg = cfg_scale
        self.sampler = sampler
        self.device = device
        self.neg_cond = self.inf.get_cond("")

        # Precompute the exact sigma schedule used by do_sampling (no denoise trimming and no trailing 0)
        sampling = self.inf.sd3.model.model_sampling
        start = sampling.timestep(sampling.sigma_max)
        end = sampling.timestep(sampling.sigma_min)
        timesteps = torch.linspace(start, end, steps)
        sigs = []
        for x in range(len(timesteps)):
            ts = timesteps[x]
            sigs.append(sampling.sigma(ts))
        self.sigmas = torch.tensor(sigs, dtype=torch.float32)  # shape: [steps]

    @torch.no_grad()
    def _run_once(self, prompt: str, seed: int, width: int, height: int, wrapper=None,
                  tag: Optional[str] = None, save_dir: Optional[str] = None) -> Image.Image:
        latent = self.inf.get_empty_latent(1, width, height, seed, device="cuda")
        cond = self.inf.get_cond(prompt)
        ncond = self.neg_cond

        if wrapper is not None:
            def _fake_cfg(*args, **kwargs):
                return wrapper
            _sdi.CFGDenoiser = _fake_cfg
            _sdm.CFGDenoiser = _fake_cfg

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
            if wrapper is not None:
                if original_cfg_infer is not None:
                    _sdi.CFGDenoiser = original_cfg_infer
                if original_cfg_impl is not None:
                    _sdm.CFGDenoiser = original_cfg_impl

        img = self.inf.vae_decode(sampled_latent)
        if save_dir and tag:
            os.makedirs(save_dir, exist_ok=True)
            img.save(os.path.join(save_dir, f"{tag}.png"))
        return img

    def train(
        self,
        bank: PolicyBank,
        reward_fn: Callable[[str, Image.Image], float],
        cfg: TrainConfig,
    ):
        prompts = list(cfg.prompts)
        seeds   = list(cfg.seeds)

        # materialize/size policies for current latent shape
        with torch.no_grad():
            z0 = self.inf.get_empty_latent(1, cfg.width, cfg.height, seeds[0], device="cuda")
        bank.reset_policies(z0, cfg.schedule)

        opt = torch.optim.AdamW(bank.parameters(), lr=cfg.lr, betas=(0.9, 0.999), weight_decay=1e-4)

        if cfg.resume_from and os.path.exists(cfg.resume_from):
            state = torch.load(cfg.resume_from, map_location="cpu")
            bank.load_state_dict(state["policy_bank"])
            print(f"Resumed policy_bank from {cfg.resume_from}")

        for epoch in range(1, cfg.num_epochs + 1):
            for t in cfg.schedule:
                for i, pr in enumerate(prompts):
                    for j, sd in enumerate(seeds):
                        tag = f"ep{epoch:02d}_t{t:03d}_p{i:02d}_s{j:02d}_ref"
                        save_dir = (cfg.out_dir if epoch % cfg.save_every == 0 else None)
                        img_ref = self._run_once(pr, sd, cfg.width, cfg.height, wrapper=None, save_dir=save_dir, tag=tag)
                        r_ref = float(reward_fn(pr, img_ref))

                        # Collect a group of (logp, reward)
                        logps: List[torch.Tensor] = []
                        rewards: List[float] = []
                        advantages: List[float] = []
                        for g in range(cfg.group_size):
                            generator = torch.Generator(device=bank.device)
                            generator.manual_seed(hash((epoch, t, i, j, g)) & 0xFFFFFFFF)

                            base = _sdm.CFGDenoiser(self.inf.sd3.model)
                            wrapper = GRPODenoiserWrapper(
                                base_denoiser=base,
                                bank=bank,
                                schedule=(t,),                # in training, train policies for different t separately
                                cfg_scale=self.cfg,
                                sigmas=self.sigmas,           # pass precomputed schedule
                                total_steps=self.steps,
                                generator=generator,
                            )
                            tag = f"ep{epoch:02d}_t{t:03d}_p{i:02d}_s{j:02d}_g{g:02d}"
                            img = self._run_once(pr, sd, cfg.width, cfg.height,
                                                 wrapper=wrapper,
                                                 tag=tag,
                                                 save_dir=(cfg.out_dir if epoch % cfg.save_every == 0 else None))
                            r = float(reward_fn(pr, img))
                            rewards.append(r)
                            logps.append(wrapper.logged_logp)
                            advantages.append(r - r_ref)

                        # Group-normalised advantages
                        normalized_advantages = torch.tensor(advantages, device=bank.device, dtype=torch.float32)
                        normalized_advantages = (normalized_advantages - normalized_advantages.mean()) / normalized_advantages.std(unbiased=False).clamp_min(1e-6)  # (G,)

                        logp_tensor = torch.stack(logps, dim=0)  # (G,)
                        action_dim = self.bank.policy(int(t)).actor.net[-1].out_features // 2 # action dim
                        loss = -(normalized_advantages.detach() * (logp_tensor / max(1, action_dim))).mean()
                        logger({
                            "epoch": epoch,
                            "t": t,
                            "prompt_idx": i,
                            "seed_idx": j,
                            "rewards": rewards,
                            "advantages": advantages.tolist(),
                            "normalized_advantages": normalized_advantages.tolist(),
                            "logps": logp_tensor.tolist(),
                            "loss": loss.item(),
                        }, f"{cfg.out_dir}/training_log.csv")

                        opt.zero_grad(set_to_none=True)
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(bank.parameters(), cfg.max_grad_norm)
                        opt.step()
                        

            if epoch % cfg.save_every == 0:
                os.makedirs(cfg.out_dir, exist_ok=True)
                ckpt = {"policy_bank": bank.state_dict(), "schedule": cfg.schedule, "epoch": epoch}
                torch.save(ckpt, os.path.join(cfg.out_dir, f"policy_bank_epoch_{epoch:02d}.pt"))
                print(f"[GRPO] Saved checkpoint for epoch {epoch}")

        print("GRPO training complete.")