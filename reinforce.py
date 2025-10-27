# reinforce.py
# Per-timestep, single-intervention REINFORCE for SD3.5
# Implements both 'latent_delta' and 'basis_delta' nudges with a learned policy per step t.

from dataclasses import dataclass
from typing import Callable, Dict, Optional, Sequence, Tuple
import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import sd3_impls as _sd3

# ------------------ utils ------------------

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
      - last action summary (2): ||a||, sign-mean (optional; zeros for single-action runs)
    -> ~26 dims, padded to state_dim.
    """
    def __init__(self, state_dim: int = 128, t_emb_dim: int = 16, device: str = "cuda"):
        self.state_dim = state_dim
        self.t_emb_dim = t_emb_dim
        self.device = device

    def build(self, latent: torch.Tensor, t_idx: int, T: int, cfg_scale: float) -> torch.Tensor:
        z = latent
        with torch.no_grad():
            stats = torch.stack([
                z.mean(), z.std(), z.abs().mean(),
                z.norm(p=2) / z.numel()**0.5,
                z.min(), z.max(),
            ]).to(self.device, dtype=torch.float32)
            t_frac = float(t_idx) / max(1, T-1)
            t_emb = sinusoidal_embed(t_frac, dim=self.t_emb_dim, device=self.device)
            extra = torch.tensor([cfg_scale, t_frac], device=self.device, dtype=torch.float32)
            s = torch.cat([stats, t_emb, extra], dim=0)
        if s.numel() < self.state_dim:
            s = F.pad(s, (0, self.state_dim - s.numel()))
        else:
            s = s[:self.state_dim]
        return s  # (state_dim,)

# ------------------ per-step policy & critic ------------------

class StepPolicy(nn.Module):
    """
    A tiny Gaussian policy head for ONE timestep t.
    Modes:
      - 'latent_delta' : action a ∈ R^D (D flattened latent)
      - 'basis_delta'  : action u ∈ R^k, delta z = B @ u
    """
    def __init__(
        self,
        mode: str,
        state_dim: int,
        action_dim: int,          # k for basis; ignored for latent until we know D
        init_log_std: float = -1.0,
        alpha: float = 0.02,
        device: str = "cuda",
    ):
        super().__init__()
        assert mode in ("latent_delta", "basis_delta")
        self.mode = mode
        self.alpha = alpha
        self.device = device

        self.actor = TinyMLP(state_dim, 2*action_dim).to(device)  # resized later for latent_delta
        with torch.no_grad():
            last = [m for m in self.actor.modules() if isinstance(m, nn.Linear)][-1]
            A = last.out_features // 2
            last.bias[A:] = init_log_std

        self.critic = TinyMLP(state_dim, 1).to(device)

        # will be set when we see the first latent
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

# ------------------ policy bank (exposes get_policy) ------------------

class PolicyBank:
    """
    Holds an independent StepPolicy (+critic) for each timestep t.
    This gives you the conceptual get_policy(latent_t, t).
    """
    def __init__(
        self,
        mode: str = "basis_delta",
        state_dim: int = 128,
        action_dim_basis: int = 64,
        alpha: float = 0.02,
        device: str = "cuda",
    ):
        self.mode = mode
        self.state_dim = state_dim
        self.action_dim_basis = action_dim_basis
        self.alpha = alpha
        self.device = device
        self.bank: Dict[int, StepPolicy] = {}
        self.state_builder = StateBuilder(state_dim=state_dim, device=device)

    def policy(self, t: int) -> StepPolicy:
        if t not in self.bank:
            self.bank[t] = StepPolicy(
                mode=self.mode,
                state_dim=self.state_dim,
                action_dim=(self.action_dim_basis if self.mode == "basis_delta" else 2),  # temp; resized later
                alpha=self.alpha,
                device=self.device,
            )
        return self.bank[t]

    def get_policy(self, latent_t: torch.Tensor, t: int, T: int, cfg_scale: float):
        """
        Returns callable (action, logp, value, state) for this t.
        This matches your desired interface conceptually.
        """
        pol = self.policy(t)
        pol.ensure_shapes(latent_t, action_dim_basis=self.action_dim_basis)
        s_t = self.state_builder.build(latent_t, t, T, cfg_scale).to(self.device)
        a_t, logp_t = pol.act(s_t)
        v_t = pol.value(s_t)
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

    def parameters(self):
        # collect all actor+critic params
        params = []
        for p in self.bank.values():
            params += list(p.actor.parameters()) + list(p.critic.parameters())
        return params
    
    def load_weights(self, ckpt_path):
        ckpt = torch.load(ckpt_path, map_location="cpu")
        self.schedule = ckpt.get("schedule", None)
        self.load_state_dict(ckpt["policy_bank"])
        print(f"Loaded weights from {ckpt_path}")

    def reset_policies(self, latent: torch.Tensor, schedule: Sequence[int]):
        for t in schedule:
            pol = self.policy(t)
            pol.ensure_shapes(latent, action_dim_basis=self.action_dim_basis)


# ------------------ denoiser wrapper for a single step ------------------

class SingleStepDenoiserWrapper:
    """
    Wraps CFGDenoiser so ONLY step t* applies a policy nudge to x.
    Tracks calls via an internal counter (t_idx). Mirrors CFGDenoiser.forward signature.
    """
    def __init__(self, base_denoiser, bank: PolicyBank, target_t: int, cfg_scale: float, total_steps: int):
        self.base = base_denoiser
        self.bank = bank
        self.t_star = target_t
        self.cfg_scale = cfg_scale
        self.T = int(total_steps)

        # logging for RL update
        self.logged = False
        self.logged_state = None
        self.logged_logp = None
        self.logged_value = None
        self.logged_action = None

        self.t_idx = 0  # step counter

    def forward(self, x, timestep, cond, uncond, cond_scale, save_tensors_path=None, **kwargs):
        # Apply action BEFORE the model call, at the chosen step
        if self.t_idx == self.t_star and not self.logged:
            a_t, logp_t, v_t, s_t = self.bank.get_policy(x, self.t_idx, self.T, self.cfg_scale)
            x = self.bank.apply_action(x, a_t, self.t_idx)

            # log for learning
            self.logged = True
            self.logged_state = s_t
            self.logged_logp = logp_t
            self.logged_value = v_t
            self.logged_action = a_t

        # Call the real denoiser
        out = self.base.forward(
            x, timestep, cond, uncond, cond_scale,
            save_tensors_path=save_tensors_path, **kwargs
        )

        self.t_idx += 1
        return out

    # Allow being called like a function, just like a torch.nn.Module
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

# ------------------ trainer ------------------

@dataclass
class TrainConfig:
    schedule: Tuple[int, ...] = (10, 20, 35)  # which steps to train
    num_epochs: int = 3                        # sweeps over all t
    iters_per_t: int = 1                       # how many updates per t per epoch
    lr: float = 3e-4
    value_coef: float = 0.5
    entropy_coef: float = 0.0                  # set >0 if you add explicit entropy tracking
    max_grad_norm: float = 1.0
    save_every: int = 1
    out_dir: str = "outputs/rl_per_timestep"
    resume_from: Optional[str] = None

class SingleStepTrainer:
    """
    Baseline: run once with NO intervention -> R_base.
    For each t in schedule: wrap denoiser to act ONLY at t, run once, get R_t, update ONLY π_t, V_t.
    Repeat over epochs (coordinate ascent over steps).
    """
    def __init__(self, inferencer, prompt: str, width: int, height: int, steps: int, cfg: float, sampler: str, seed: int, device: str = "cuda"):
        self.inf = inferencer
        self.prompt = prompt
        self.width = width
        self.height = height
        self.steps = steps
        self.cfg = cfg
        self.sampler = sampler
        self.seed = seed
        self.device = device

        self.cond = self.inf.get_cond(prompt)
        self.neg_cond = self.inf.get_cond("")
        with torch.no_grad():
            self.z0 = self.inf.get_empty_latent(1, width, height, seed, device="cuda")

    @torch.no_grad()
    def _run_once(self, wrapper=None, tag=None, save_dir=None) -> Image.Image:
        # Prepare inputs
        latent = self.z0.clone().to("cuda")
        cond = self.inf.fix_cond(self.cond)
        ncond = self.inf.fix_cond(self.neg_cond)

        
        original_cfg = _sd3.CFGDenoiser

        if wrapper is None:
            # vanilla
            pass
        else:
            # monkey-patch
            def _fake_cfg(*args, **kwargs):
                return wrapper
            _sd3.CFGDenoiser = _fake_cfg

        try:
            sampled_latent = self.inf.do_sampling(
                latent=latent,
                seed=self.seed,
                conditioning=cond,
                neg_cond=ncond,
                steps=self.steps,
                cfg_scale=self.cfg,
                sampler=self.sampler,
                controlnet_cond=None,
                denoise=1.0,
                skip_layer_config={},
                save_tensors_path=None,
                experiment_setting="rl_single_step",
            )
        finally:
            _sd3.CFGDenoiser = original_cfg

        img = self.inf.vae_decode(sampled_latent)
        if save_dir and tag:
            os.makedirs(save_dir, exist_ok=True)
            img.save(os.path.join(save_dir, f"{tag}.png"))
        return img

    def train(self, bank: PolicyBank, reward_fn: Callable[[Image.Image], float], cfg: TrainConfig):
        # One optimizer over all per-t heads:
        if cfg.resume_from is not None:
            bank.load_weights(cfg.resume_from)
            print(f"Resumed training from {cfg.resume_from}")
        else:
            bank.reset_policies(self.z0, cfg.schedule)
            print(f"Reset policies for new training")

        opt = torch.optim.AdamW(bank.parameters(), lr=cfg.lr, betas=(0.9, 0.999), weight_decay=1e-4)

        for epoch in range(1, cfg.num_epochs + 1):
            # 1) Baseline (no intervention)
            img_base = self._run_once(wrapper=None, tag=f"epoch{epoch:02d}_base", save_dir=(cfg.out_dir if (epoch % cfg.save_every)==0 else None))
            R_base = float(reward_fn(img_base))

            for t in cfg.schedule:
                for _ in range(cfg.iters_per_t):
                    # 2) Single-step intervention at t
                    from sd3_impls import CFGDenoiser
                    base = CFGDenoiser(self.inf.sd3.model)  # constructor only needs model in your impl
                    wrapper = SingleStepDenoiserWrapper(
                        base_denoiser=base,
                        bank=bank,
                        target_t=t,
                        cfg_scale=self.cfg,
                        total_steps=self.steps,   # <— important now
                    )
                    img_t = self._run_once(wrapper=wrapper, tag=f"epoch{epoch:02d}_t{t:03d}", save_dir=(cfg.out_dir if (epoch % cfg.save_every)==0 else None))
                    R_t = float(reward_fn(img_t))

                    # 3) If action didn’t fire (shouldn’t happen), skip
                    if wrapper.logged_state is None:
                        continue

                    s_t = wrapper.logged_state.detach()
                    logp_t = wrapper.logged_logp
                    V_t = wrapper.logged_value
                    # Advantage: use strong control variate
                    A_t = torch.tensor([R_t - R_base], device=bank.device, dtype=torch.float32) - V_t.detach()

                    policy_loss = -(A_t * logp_t).mean()
                    value_loss = cfg.value_coef * ( (R_t - R_base) - V_t ).pow(2).mean()
                    loss = policy_loss + value_loss

                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(bank.parameters(), cfg.max_grad_norm)
                    opt.step()

            print(f"[epoch {epoch}/{cfg.num_epochs}] R_base={R_base:.4f}")

            # Save checkpoint
            if epoch % cfg.save_every == 0:
                os.makedirs(cfg.out_dir, exist_ok=True)
                ckpt_path = os.path.join(cfg.out_dir, f"policy_bank_epoch_{epoch:02d}.pt")
                torch.save({
                    "policy_bank": bank.state_dict(),
                    "schedule": cfg.schedule,
                    "epoch": epoch,
                }, ckpt_path)
                print(f"Saved checkpoint to {ckpt_path}")

        print("Training complete.")

# ------------------ quick usage ------------------
# After inferencer.load(...):
#
#   from reinforce import PolicyBank, SingleStepTrainer, TrainConfig
#
#   bank = PolicyBank(mode="basis_delta", action_dim_basis=64, alpha=0.02, device="cuda")
#   trainer = SingleStepTrainer(
#       inferencer=inferencer,
#       prompt=PROMPT,
#       width=WIDTH, height=HEIGHT,
#       steps=_steps, cfg=_cfg, sampler=_sampler,
#       seed=SEED, device="cuda"
#   )
#
#   def reward_fn(img: Image.Image) -> float:
#       return score_image(img)  # your black-box scorer
#
#   cfg = TrainConfig(schedule=(8, 12, 18, 24, 30), num_epochs=3, iters_per_t=2, save_every=1,
#                     out_dir="outputs/rl_per_timestep")
#   trainer.train(bank, reward_fn, cfg)
#
# You can also call: a_t, logp, V, s = bank.get_policy(latent_t, t, T, cfg_scale)