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

def logger(d: str, filename: str):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, "a") as f:
        f.write(str(d) + "\n")

def save_tensor(t: torch.Tensor, filename: str):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    if isinstance(t, torch.Tensor):
        t = t.detach().cpu()
    torch.save(t, filename)

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
        action_alpha: float = 0.02,
        hidden: int = 256,
        device: str = "cuda",
        out_dir: str = "outputs/grpo",
        save_tensor_logs: bool = False
    ):
        super().__init__()
        assert mode in ("latent_delta", "basis_delta")
        self.mode = mode
        self.action_alpha = action_alpha
        self.hidden = hidden
        self.device = device
        self.out_dir = out_dir

        # lazily built after we see a latent (to know D)
        self.actor: nn.Sequential = None  # type: ignore
        self.actor_in_dim: Optional[int] = None
        self.action_dim_basis = action_dim_basis
        self.init_log_std = init_log_std

        # for basis mode + size tracking
        self.latent_dim: Optional[int] = None
        self.log_idx = 0
        self.save_tensor_logs = save_tensor_logs

        self.register_buffer("basis", torch.empty(0), persistent=True)

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
    def sample(self, state: torch.Tensor, generator: Optional[torch.Generator] = None, tag: Optional[str] = None):

        params = self.actor(state)  # (2*A,)
        A = params.numel() // 2
        mu, log_std = params[:A], params[A:]
        std = log_std.exp().clamp(min=1e-5, max=1)
        mu = mu.clamp(min=-1, max=1)

        if generator is None:
            eps = torch.randn_like(std)
        else:
            eps = torch.randn(std.shape, device=std.device, dtype=std.dtype, generator=generator)

        # save_tensor(eps, f"{self.out_dir}/tensors/policy_noise_{self.log_idx}.pt")
        self.log_idx += 1

        a = mu + std * eps # ~ N(mu, std^2)
        # log prob (sum over dims)
        logp = (-0.5 * (((a - mu) / std) ** 2 + 2 * log_std + math.log(2 * math.pi))).sum(-1)

        if self.save_tensor_logs:
            save_tensor(log_std, f"{self.out_dir}/tensors/policy_logstd_{tag}.pt")
            save_tensor(mu, f"{self.out_dir}/tensors/policy_mu_{tag}.pt")
            save_tensor(eps, f"{self.out_dir}/tensors/policy_noise_{tag}.pt")
        

        return a.detach(), logp  # detach action (we only backprop through logp)


# ------------------ policy bank ------------------

class PolicyBank(nn.Module):
    """
    Independent StepPolicy for each t.
    """
    def __init__(
        self,
        mode: str = "basis_delta",
        action_dim_basis: int = 64,
        action_alpha: float = 1,
        state_alpha: float = 0.02,
        hidden: int = 256,
        device: str = "cuda",
        out_dir: str = "outputs/grpo",
        save_tensor_logs: bool = False,
        latent_encoding_dim: int = 128,
        cond_encoding_dim: int = 32,
    ):
        super().__init__()
        self.mode = mode
        self.action_dim_basis = action_dim_basis
        self.action_alpha = action_alpha
        self.state_alpha = state_alpha
        self.hidden = hidden
        self.device = device
        self.out_dir = out_dir
        self.bank = nn.ModuleDict()  # keyed by str(t)
        self.save_tensor_logs = save_tensor_logs
        self.latent_encoding_dim = latent_encoding_dim
        self.cond_encoding_dim = cond_encoding_dim
        self.register_buffer("cond_basis", torch.empty(0), persistent=True)
        
    def policy(self, t: int) -> StepPolicy:
        key = str(int(t))
        if key not in self.bank:
            self.bank[key] = StepPolicy(
                mode=self.mode,
                action_dim_basis=self.action_dim_basis,
                action_alpha=self.action_alpha,
                hidden=self.hidden,
                device=self.device,
                out_dir=self.out_dir,
                save_tensor_logs=self.save_tensor_logs,
            )
        return self.bank[key]
    
    def ensure_cond_basis(self, D_in: int):
        if self.cond_basis.numel() == 0 or self.cond_basis.shape[0] != D_in or self.cond_basis.shape[1] != self.cond_encoding_dim:
            self.cond_basis = orthonormal_basis(D_in, self.cond_encoding_dim, device=self.device, dtype=torch.float32)

    def _encode_cond(self, cond: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        seq, pooled = None, None
        if isinstance(cond, tuple) and len(cond) == 2:
            seq, pooled = cond
        elif isinstance(cond, dict):
            seq = cond.get("c_crossattn", None)
            pooled = cond.get("y", None)
        else:
            raise TypeError(f"Unsupported cond type: {type(cond)}")
        
        v = pooled if torch.is_tensor(pooled) else (seq.mean(dim=-2) if seq.dim() >= 2 else seq)
        v = v.detach().to(self.device, dtype=torch.float32).flatten()
        self.ensure_cond_basis(v.numel())
        enc = self.cond_basis.T @ v
        return enc / (enc.norm(p=2) + 1e-8)

    @torch.no_grad()
    def _normalize(self, t: torch.Tensor) -> torch.Tensor:
        return (t - t.mean()) / (t.std(unbiased=False).clamp_min(1e-6))

    @torch.no_grad()
    def _encode_latent(self, latent: torch.Tensor) -> torch.Tensor:
        # placeholder for possible future encoding of latent + cond
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
        # target elements, keep aspect, and do not upsample
        scale = min(1.0, math.sqrt(self.latent_encoding_dim / max(1, H * W)))
        h_new = max(1, int(H * scale))
        w_new = max(1, int(W * scale))

        # downsample
        x = F.adaptive_avg_pool2d(x.unsqueeze(0).unsqueeze(0), output_size=(h_new, w_new)) \
                .squeeze(0).squeeze(0)   # (h_new, w_new)

        return self._normalize(x.flatten()) 

    def get_state(self, latent_t: torch.Tensor, cond: Tuple[torch.Tensor, torch.Tensor], t: int, sigma_t: float, cfg_scale: float, generator: Optional[torch.Generator] = None) -> torch.Tensor:
        if generator is not None:
            _latent = latent_t.detach() + self.state_alpha * torch.randn(latent_t.shape, device=latent_t.device, dtype=latent_t.dtype, generator=generator)
            if self.save_tensor_logs:
                save_tensor(_latent, f"{self.out_dir}/tensors/noisy_latent_t{int(sigma_t*1000):03d}.pt")
        else:
            _latent = latent_t.detach()

        latent_enc = self._encode_latent(_latent)
        cond_enc = self._encode_cond(cond)

        # add log-sigma and cfg
        sigma_feat = torch.tensor([math.log(max(1e-8, float(sigma_t)))],
                                  device=self.device, dtype=torch.float32)
        cfg_feat = torch.tensor([float(cfg_scale)], device=self.device, dtype=torch.float32)

        s_t = torch.cat([latent_enc, cond_enc, sigma_feat, cfg_feat], dim=0)

        if self.save_tensor_logs:
            save_tensor(s_t, f"{self.out_dir}/tensors/state_t{int(sigma_t*1000):03d}.pt")
        
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

        D = delta.numel()
        target_l2 = math.sqrt(D) * pol.action_alpha
        delta = delta.float()
        delta = delta * (target_l2 / delta.norm(p=2).clamp_min(1e-8))

        return (latent + delta).to(latent.dtype)

    def reset_policies(self, latent: torch.Tensor, cond: Tuple[torch.Tensor, torch.Tensor], schedule: Sequence[int]):
        # force construction/sizing for initial latent shape
        s_t = self.get_state(latent, cond, sigma_t=1.0, cfg_scale=1.0).to(self.device)
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
        out_dir: str = "outputs/grpo",
        save_tensor_logs: bool = False,
        tag: Optional[str] = None,
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
        self.out_dir = out_dir
        self.save_tensor_logs = save_tensor_logs

        self.generator = generator or torch.Generator(device=bank.device)
        if not self.generator.initial_seed():
            self.generator.manual_seed(torch.seed())

        self.tag = tag

    def forward(self, x, timestep, cond, uncond, cond_scale, save_tensors_path=None, **kwargs):
        if (not self._acted) and (self.t_idx in self.schedule):
            sigma_t = float(self.sigmas[min(self.t_idx, len(self.sigmas) - 1)])
            x_detach = x.detach()
            with torch.enable_grad():
                s_t = self.bank.get_state(x_detach, cond, self.t_idx, sigma_t, self.cfg_scale, generator=self.generator)
                a_t, logp_t = self.bank.policy(self.t_idx).sample(s_t, generator=self.generator, tag=self.tag)

            stats_before = {
                "min": x_detach.min().item(),
                "max": x_detach.max().item(),
                "mean": x_detach.mean().item(),
                "std": x_detach.std().item(),
            }
            x = self.bank.apply_action(x_detach, a_t.detach(), self.t_idx)
            stats_after = {
                "min": x.min().item(),
                "max": x.max().item(),
                "mean": x.mean().item(),
                "std": x.std().item(),
            }
            self.logged_logp = logp_t
            self._acted = True
            logger({
                "t_idx": self.t_idx,
                "sigma_t": sigma_t,
                "cfg_scale": self.cfg_scale,
                "action_norm": a_t.norm().item(),
                "logp": logp_t.item(),
                "stats_before": stats_before,
                "stats_after": stats_after,
            }, f"{self.out_dir}/logs/policy_log_{self.tag}.log")
            
            # save latent and action for debugging
            if self.save_tensor_logs:
                save_tensor(x, f"{self.out_dir}/tensors/latent_{self.tag}.pt")
                save_tensor(a_t, f"{self.out_dir}/tensors/action_{self.tag}.pt")

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
    out_dir: str = "outputs/grpo"
    resume_from: Optional[str] = None

    # Batch prompts/seeds, resolution
    prompts: Sequence[str] = ("a photo of a cat",)
    seeds: Sequence[int] = (23,)
    width: int = 1024
    height: int = 1024
    save_tensor_logs: bool = False


class GRPOTrainer:
    """
    For each (prompt, seed, t): sample G actions (group), compute rewards, normalise advantages,
    loss = -mean(A_i * logpi_i). No critic.
    """
    def __init__(self, inferencer, steps: int, cfg_scale: float, sampler: str, device: str = "cuda", out_dir: str = "outputs/grpo"):
        self.inf = inferencer
        self.steps = steps
        self.cfg = cfg_scale
        self.sampler = sampler
        self.device = device
        self.out_dir = out_dir
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
        # save_tensor(latent, f"outputs/grpo_mock_scorer/initial_latent_{tag}.pt")
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
        generate_seeds = len(seeds) == 0

        # materialize/size policies for current latent shape
        with torch.no_grad():
            z0 = self.inf.get_empty_latent(1, cfg.width, cfg.height, 23, device="cuda")
        bank.reset_policies(z0, self.neg_cond, cfg.schedule)

        opt = torch.optim.AdamW(bank.parameters(), lr=cfg.lr, betas=(0.9, 0.999), weight_decay=1e-4)

        if cfg.resume_from and os.path.exists(cfg.resume_from):
            state = torch.load(cfg.resume_from, map_location="cpu")
            bank.load_state_dict(state["policy_bank"])
            print(f"Resumed policy_bank from {cfg.resume_from}")

        print("Starting GRPO training...")
        print("Prompts:", prompts)
        print("Seeds:", seeds if not generate_seeds else "will generate per iteration")
        print("Training configuration:")
        print(cfg)

        for epoch in range(1, cfg.num_epochs + 1):
            for t in cfg.schedule:
                for i, pr in enumerate(prompts):
                    
                    if generate_seeds:
                        seeds = [torch.randint(0, 2**30, (1,)).item()] # generate a seed

                    for j, sd in enumerate(seeds):
                        tag = f"ep{epoch:02d}_t{t:03d}_p{i:02d}_s{j:02d}_ref"
                        save_dir = (os.path.join(cfg.out_dir, 'imgs') if epoch % cfg.save_every == 0 else None)
                        img_ref = self._run_once(pr, sd, cfg.width, cfg.height, wrapper=None, save_dir=save_dir, tag=tag)
                        r_ref = float(reward_fn(pr, img_ref))

                        # Collect a group of (logp, reward)
                        logps: List[torch.Tensor] = []
                        rewards: List[float] = []
                        advantages: List[float] = []
                        for g in range(cfg.group_size):
                            generator = torch.Generator(device=bank.device)
                            generator.manual_seed(hash((epoch, t, i, j, g)) & 0xFFFFFFFF)
                            
                            tag = f"ep{epoch:02d}_t{t:03d}_p{i:02d}_s{j:02d}_g{g:02d}"

                            base = _sdm.CFGDenoiser(self.inf.sd3.model)
                            wrapper = GRPODenoiserWrapper(
                                base_denoiser=base,
                                bank=bank,
                                schedule=(t,),                # in training, train policies for different t separately
                                cfg_scale=self.cfg,
                                sigmas=self.sigmas,           # pass precomputed schedule
                                total_steps=self.steps,
                                generator=generator,
                                out_dir=cfg.out_dir,
                                save_tensor_logs=cfg.save_tensor_logs,
                                tag=tag,
                            )
                            img = self._run_once(pr, sd, cfg.width, cfg.height,
                                                 wrapper=wrapper,
                                                 tag=tag,
                                                 save_dir=(os.path.join(cfg.out_dir, 'imgs') if epoch % cfg.save_every == 0 else None))
                            r = float(reward_fn(pr, img))
                            rewards.append(r)
                            logps.append(wrapper.logged_logp)
                            advantages.append(r - r_ref)

                        # Group-normalised advantages
                        normalized_advantages = torch.tensor(advantages, device=bank.device, dtype=torch.float32)
                        normalized_advantages = (normalized_advantages - normalized_advantages.mean()) / normalized_advantages.std(unbiased=False).clamp_min(1e-6)  # (G,)

                        normalized_rewards = torch.tensor(rewards, device=bank.device, dtype=torch.float32)
                        normalized_rewards = (normalized_rewards - normalized_rewards.mean()) / normalized_rewards.std(unbiased=False).clamp_min(1e-6)  # (G,)

                        logp_tensor = torch.stack(logps, dim=0)  # (G,)
                        last_linear = [m for m in bank.policy(int(t)).actor.modules() if isinstance(m, torch.nn.Linear)][-1]
                        action_dim = last_linear.out_features // 2
                        loss = -(normalized_rewards.detach() * (logp_tensor / max(1, action_dim))).mean()
                        logger({
                            "epoch": epoch,
                            "t": t,
                            "prompt_idx": i,
                            "seed_idx": j,
                            'ref_reward': r_ref,
                            "rewards": rewards,
                            "advantages": advantages,
                            "normalized_advantages": normalized_advantages.tolist(),
                            "normalized_rewards": normalized_rewards.tolist(),
                            "logps": logp_tensor.tolist(),
                            "loss": loss.item(),
                        }, f"{cfg.out_dir}/logs/training_log_group.log")

                        opt.zero_grad(set_to_none=True)
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(bank.parameters(), cfg.max_grad_norm)
                        opt.step()
                        

            if epoch % cfg.save_every == 0:
                os.makedirs(cfg.out_dir, exist_ok=True)
                ckpt = {"policy_bank": bank.state_dict(), "schedule": cfg.schedule, "epoch": epoch}
                save_tensor(ckpt, os.path.join(cfg.out_dir, f"checkpoints/policy_bank_epoch_{epoch:02d}.pt"))
                print(f"[GRPO] Saved checkpoint for epoch {epoch}")

        print("GRPO training complete.")