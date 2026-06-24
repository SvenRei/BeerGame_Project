"""
draco_v4.py -- DRACO v4 (REVIEW-CORRECTED). Single self-contained file, no v1/v2/v3
inheritance.

This revision keeps everything that was verified correct (env wrapper, encoders,
both actor heads, message head, rollout buffer, the causal demand-belief ELBO with
FIX 3) VERBATIM, and replaces the value/critic machinery with the corrected version
that resolves the issues found in review:

  CORR-A -- SCALAR CRITIC (was an inert "distributional" critic).  The previous
    critic regressed all 32 quantiles to the SCALAR lambda-return, which collapses
    the quantiles to a point -- i.e. it was a mean critic wearing 32 heads, and its
    CVaR readout was meaningless. The risk pathway (risk_eta) reweights EMPIRICAL
    trajectory returns and never touched the critic, so nothing is lost by going
    scalar: it is faster, has fewer parameters, and is correct. (If a genuine
    distributional/CVaR-from-critic contribution is wanted later, swap in a critic
    with a proper distributional Bellman target -- that is a separate change.)

  CORR-B -- VALUE-TARGET NORMALIZATION replaces the hand-tuned reward_scale.  Because
    PPO advantages are normalized, reward_scale never changed the actor's gradient
    direction; it only rescaled the critic's regression target so the head could
    reach it. A fixed scale is fragile across the DR curriculum (episode cost varies
    several-fold). v4-fixed keeps rewards in RAW units and normalizes the value
    TARGET with a running mean/std (PopArt-lite): the critic predicts a unit-scale
    value, which is de-normalized for GAE. reward_scale is no longer read.

  CORR-C -- NO POLYAK TARGET CRITIC.  A target net stabilizes a *bootstrapped TD*
    target; the regression target here is the lambda-return (essentially MC), so a
    target net only made the GAE baseline lag by ~1/tau updates. v4-fixed uses the
    online critic for both the baseline and the regression target (standard PPO).
    polyak_tau is no longer read.

  KEPT: FIX 2 (critic target = GAE lambda-return = V + adv) and FIX 3 (the encoder
    reconstructs RAW demand, no /100, so the belief actually encodes the regime --
    this is load-bearing for the structured head, whose S scales with d_hat).

  ORDERING SAFETY VALVES (both OFF by default -- the structured head is the primary
    ordering fix; these are documented tools, not default behavior):
      order_cap_coef : penalizes desired order (S - IP) exceeding max_order, which
        restores a gradient in the otherwise-flat saturated region. Turn on (~0.05)
        only if you see Diag orders pegging the cap and S running away.
      corr_l2_coef   : L2 on the structured head's bounded state-correction, pulling
        S toward the clean demand-grounded base (lead*d_hat + safety). Turn on
        (~1e-3) for less bullwhip / cleaner interpretation. OOD level-tracking still
        comes from d_hat (the base), not corr, so this does not block it.

Self-contained: only external dependency is the environment.
"""
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

from envs.beer_game_env import BeerGameParallelEnv


# Chain adjacency = neighbour-only comm matrix, ROW-NORMALISED (receiver gets the
# MEAN of its neighbours' messages). DIAL routing is incoming = ADJ @ messages.
#   retailer(0) <-> wholesaler(1) <-> distributor(2) <-> manufacturer(3)
ADJ = torch.tensor([
    [0.0, 1.0, 0.0, 0.0],
    [0.5, 0.0, 0.5, 0.0],
    [0.0, 0.5, 0.0, 0.5],
    [0.0, 0.0, 1.0, 0.0],
])


def _inv_softplus(y):
    return float(math.log(math.expm1(y))) if y < 20 else float(y)


def _causal_mask(T, device):
    return torch.triu(torch.full((T, T), float("-inf"), device=device), diagonal=1)


# ==============================================================================
# Module E -- demand randomization (training only; env file untouched)
# ==============================================================================
class DemandRandomizedBeerGame(BeerGameParallelEnv):
    """Per episode: lambda ~ U[lo,hi] and, with prob p_shift, one mid-episode level
    shift. Perturbs ONLY demand_type 'poisson' (training); black_swan/extreme_chaos
    pass straight through so the OOD benchmark is never altered."""
    def __init__(self, config, lam_lo=4.0, lam_hi=16.0, p_shift=0.5, shift_scale=2.0):
        super().__init__(config)
        self._dr = dict(lo=lam_lo, hi=lam_hi, p_shift=p_shift, scale=shift_scale)
        self._dr_rng = np.random.default_rng()
        self._dr_lambda = 8.0
        self._dr_shift_t = None
        self._dr_shift_lambda = 8.0

    def reset(self, seed=None, options=None):
        if seed is not None:
            self._dr_rng = np.random.default_rng(seed + 99991)
        self._dr_lambda = float(self._dr_rng.uniform(self._dr["lo"], self._dr["hi"]))
        if self._dr_rng.random() < self._dr["p_shift"]:
            # Guard against degenerate/short horizons: integers(lo, hi) needs hi > lo,
            # and a shift time of 0 is meaningless. For horizon=50 this is integers(12, 37).
            lo_t = max(1, self.horizon // 4)
            hi_t = max(lo_t + 1, 3 * self.horizon // 4)
            self._dr_shift_t = int(self._dr_rng.integers(lo_t, hi_t))
            factor = self._dr["scale"] if self._dr_rng.random() < 0.5 else 1.0 / self._dr["scale"]
            self._dr_shift_lambda = max(0.0, self._dr_lambda * factor)
        else:
            self._dr_shift_t = None
        return super().reset(seed=seed, options=options)

    def _roll_stochastic_demand(self, step):
        if self._config.get("demand_type") == "poisson":
            lam = self._dr_lambda
            if self._dr_shift_t is not None and step >= self._dr_shift_t:
                lam = self._dr_shift_lambda
            return self.np_random.poisson(lam)
        return super()._roll_stochastic_demand(step)


# ==============================================================================
# Module B -- ENCODERS (action-free, swappable: CRAFT transformer | GRU)
#   forward_sequence(obs[T,N,od], msg[T,N,msg]) -> mu[T,N,z], logstd[T,N,z], pred[T,N,1]
#   N is the batch dim, T is time. Both are ACTION-FREE and CAUSAL.
# ==============================================================================
class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        assert d_model % 2 == 0, "craft_dim must be even"
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))               # [1, max_len, d]

    def forward(self, x):                                          # x [B,T,d]
        assert x.size(1) <= self.pe.size(1), \
            f"sequence length {x.size(1)} exceeds craft_max_len {self.pe.size(1)}"
        return x + self.pe[:, : x.size(1)]


class _CraftBlock(nn.Module):
    """Pre-LN transformer block with causal multi-head self-attention."""
    def __init__(self, d_model, heads, ff, dropout):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, heads, dropout=dropout, batch_first=True)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(nn.Linear(d_model, ff), nn.GELU(), nn.Linear(ff, d_model))
        self.drop = nn.Dropout(dropout)

    def forward(self, x, mask):                                    # x [B,T,d]
        h = self.ln1(x)
        a, _ = self.attn(h, h, h, attn_mask=mask, need_weights=False)
        x = x + self.drop(a)
        h = self.ln2(x)
        x = x + self.drop(self.ff(h))
        return x


class CraftEncoder(nn.Module):
    """Action-free causal transformer demand-belief encoder."""
    def __init__(self, obs_dim, msg_dim, z_dim, d_model=128, heads=4, layers=2,
                 ff=256, dropout=0.0, max_len=64):
        super().__init__()
        self.z_dim = z_dim
        self.in_proj = nn.Linear(obs_dim + msg_dim, d_model)       # action-free: obs (+ incoming msg)
        self.pos = SinusoidalPositionalEncoding(d_model, max_len)
        self.blocks = nn.ModuleList([_CraftBlock(d_model, heads, ff, dropout) for _ in range(layers)])
        self.ln_f = nn.LayerNorm(d_model)
        self.mu = nn.Linear(d_model, z_dim)
        self.logstd = nn.Linear(d_model, z_dim)
        self.demand_head = nn.Sequential(nn.Linear(z_dim, d_model // 2), nn.ReLU(),
                                         nn.Linear(d_model // 2, 1))

    def forward_sequence(self, obs_seq, msg_seq):                  # [T,N,*]
        x = torch.cat([obs_seq / 100.0, msg_seq], dim=-1).transpose(0, 1).contiguous()  # [N,T,*]
        x = self.pos(self.in_proj(x))
        mask = _causal_mask(x.size(1), x.device)
        for blk in self.blocks:
            x = blk(x, mask)
        x = self.ln_f(x)                                           # [N,T,d]
        mu = self.mu(x)
        logstd = self.logstd(x).clamp(-5.0, 2.0)
        pred = self.demand_head(mu)
        return mu.transpose(0, 1), logstd.transpose(0, 1), pred.transpose(0, 1)


class GruEncoder(nn.Module):
    """Ablation baseline -- action-free GRU with the identical interface."""
    def __init__(self, obs_dim, msg_dim, z_dim, hidden=128, **_):
        super().__init__()
        self.z_dim = z_dim
        self.gru = nn.GRU(obs_dim + msg_dim, hidden)               # action-free
        self.mu = nn.Linear(hidden, z_dim)
        self.logstd = nn.Linear(hidden, z_dim)
        self.demand_head = nn.Sequential(nn.Linear(z_dim, hidden // 2), nn.ReLU(),
                                         nn.Linear(hidden // 2, 1))

    def forward_sequence(self, obs_seq, msg_seq):                  # [T,N,*]
        x = torch.cat([obs_seq / 100.0, msg_seq], dim=-1)         # [T,N,*] (T=seq, N=batch)
        out, _ = self.gru(x)                                       # [T,N,hidden]
        mu = self.mu(out)
        logstd = self.logstd(out).clamp(-5.0, 2.0)
        pred = self.demand_head(mu)
        return mu, logstd, pred


def make_encoder(kind, obs_dim, msg_dim, z_dim, cfg):
    kind = (kind or "craft").lower()
    if kind == "gru":
        return GruEncoder(obs_dim, msg_dim, z_dim, hidden=cfg.get("hidden_dim", 128))
    return CraftEncoder(obs_dim, msg_dim, z_dim,
                        d_model=cfg.get("craft_dim", 128),
                        heads=cfg.get("craft_heads", 4),
                        layers=cfg.get("craft_layers", 2),
                        ff=cfg.get("craft_ff", 256),
                        dropout=cfg.get("craft_dropout", 0.0),
                        max_len=cfg.get("craft_max_len", 64))


# ==============================================================================
# Module A -- base-stock actor   +   Module C (DIAL) -- continuous message head
# ==============================================================================
class BaseStockActor(nn.Module):
    """Heterogeneous (one per echelon). Emits a base-stock TARGET S as a Gaussian
    policy head (the ONLY PPO action). Env order = clip(S - IP, 0, max_order). The
    incoming neighbour message enters LIVE so DIAL gradients reach the sender.

    NOTE (negative control): the free MLP maps (obs, z, msg) -> S, so S can vary with
    inventory; this CANNOT extrapolate S to an unseen demand level the way the
    structured head can. Use it as the ablation, not the contender."""
    def __init__(self, obs_dim, z_dim, msg_dim, hidden, max_order,
                 s_bias_init=40.0, s_logstd_init=0.7):
        super().__init__()
        self.max_order = float(max_order)
        self.obs_enc = nn.Linear(obs_dim, hidden // 2)
        self.trunk = nn.Sequential(nn.Linear(hidden // 2 + z_dim + msg_dim, hidden), nn.ReLU())
        self.s_mu = nn.Linear(hidden, 1)
        self.s_logstd = nn.Parameter(torch.zeros(1) + s_logstd_init)
        nn.init.constant_(self.s_mu.bias, _inv_softplus(s_bias_init))
        self._last_corr = None   # MLP head has no structural correction term

    def forward(self, obs, z, incoming):
        f = self.trunk(torch.cat([F.relu(self.obs_enc(obs / 100.0)), z, incoming], dim=-1))
        s_mu = F.softplus(self.s_mu(f))
        s_std = self.s_logstd.exp().clamp(1e-3, 5.0)
        return s_mu, s_std

    @staticmethod
    def order_from_S(S, obs, max_order):
        IP = obs[..., 0:1] - obs[..., 1:2] + obs[..., 2:3]        # inv - backlog + on_order
        order = torch.clamp(S - IP, min=0.0, max=max_order)
        return order, IP


class BaseStockActorStructured(nn.Module):
    """Demand-grounded base-stock head. Instead of a free MLP mapping (obs,z,msg)->S
    (which lets S swing with state and extrapolates badly when demand jumps OOD),
    this head BAKES IN the base-stock formula:

        d_hat   = softplus(d_head(z))                 # demand estimate from the belief
        base    = softplus(log_lead) * d_hat + softplus(log_safety)
        corr    = corr_scale * tanh(corr_net([obs, z, msg]))   # small, bounded nudge
        S       = softplus(base + corr)

    Why this is the primary ordering fix:
      * S scales ~LINEARLY with an explicit demand estimate, so when demand jumps
        (e.g. 8 -> 20) S rises with it instead of staying stuck near the in-dist
        level -- this is exactly the OOD level-tracking the free MLP cannot do.
      * The free part is only `corr`, bounded to +/- corr_scale, so S can't oscillate
        wildly -> far less order variance / bullwhip.
      * d_hat is grounded by a small supervised aux loss in the trainer
        (demand_estimate vs realized demand), which only works because FIX 3 makes
        the belief encode the demand regime in the first place.

    Interface matches BaseStockActor (forward -> s_mu, s_std), plus demand_estimate(z)
    for the aux loss. order_from_S is inherited via the same staticmethod."""
    def __init__(self, obs_dim, z_dim, msg_dim, hidden, max_order,
                 s_bias_init=40.0, s_logstd_init=0.7,
                 lead_init=4.0, demand_init=8.0, corr_scale=6.0):
        super().__init__()
        self.max_order = float(max_order)
        self.corr_scale = float(corr_scale)
        self.log_lead = nn.Parameter(torch.tensor([_inv_softplus(lead_init)]))
        # safety chosen so base ~= s_bias_init at the warm-start demand:
        #   lead_init * demand_init + safety = s_bias_init -> safety = s_bias_init - lead*demand
        safety_init = max(1.0, s_bias_init - lead_init * demand_init)
        self.log_safety = nn.Parameter(torch.tensor([_inv_softplus(safety_init)]))
        # demand readout from the belief; bias so d_hat ~= demand_init at z=0
        self.d_head = nn.Linear(z_dim, 1)
        nn.init.zeros_(self.d_head.weight)
        nn.init.constant_(self.d_head.bias, _inv_softplus(demand_init))
        # small bounded state correction
        self.corr_net = nn.Sequential(
            nn.Linear(obs_dim + z_dim + msg_dim, hidden), nn.ReLU(), nn.Linear(hidden, 1))
        nn.init.zeros_(self.corr_net[-1].weight); nn.init.zeros_(self.corr_net[-1].bias)  # corr~0 at init
        self.s_logstd = nn.Parameter(torch.zeros(1) + s_logstd_init)
        self._last_corr = None   # exposed for optional corr_l2 regularization in the trainer

    def demand_estimate(self, z):
        """Belief -> non-negative demand estimate. Grounded by the trainer aux loss."""
        return F.softplus(self.d_head(z))                         # [.,1]

    def forward(self, obs, z, incoming):
        d_hat = self.demand_estimate(z)
        lead = F.softplus(self.log_lead)
        safety = F.softplus(self.log_safety)
        base = lead * d_hat + safety                              # base-stock target level
        corr = self.corr_scale * torch.tanh(
            self.corr_net(torch.cat([obs / 100.0, z, incoming], dim=-1)))
        self._last_corr = corr                                    # for optional corr_l2 penalty
        s_mu = F.softplus(base + corr)
        s_std = self.s_logstd.exp().clamp(1e-3, 5.0)
        return s_mu, s_std

    @staticmethod
    def order_from_S(S, obs, max_order):
        return BaseStockActor.order_from_S(S, obs, max_order)


def make_actor(head, obs_dim, z_dim, msg_dim, hidden, max_order, cfg):
    """actor_head: 'mlp' (free MLP S head, negative control) | 'structured' (primary)."""
    head = (head or "mlp").lower()
    common = dict(s_bias_init=cfg.get("s_bias_init", 40.0),
                  s_logstd_init=cfg.get("s_logstd_init", 0.7))
    if head == "structured":
        return BaseStockActorStructured(
            obs_dim, z_dim, msg_dim, hidden, max_order,
            lead_init=cfg.get("lead_init", 4.0),
            demand_init=cfg.get("demand_init", 8.0),
            corr_scale=cfg.get("corr_scale", 6.0), **common)
    return BaseStockActor(obs_dim, z_dim, msg_dim, hidden, max_order, **common)


class MessageHead(nn.Module):
    """DIAL message head (one per echelon). Reads the DETACHED belief z and local obs,
    emits a bounded continuous vector. NO log-prob, NOT an action: trained purely by
    the gradient flowing back from receivers' policy loss through ADJ. Reading
    z.detach() keeps the encoder clean -- DIAL shapes only this head."""
    def __init__(self, obs_dim, z_dim, msg_dim, hidden):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(obs_dim + z_dim, hidden // 2), nn.ReLU(),
                                 nn.Linear(hidden // 2, msg_dim))

    def forward(self, obs, z):                                     # z already detached by caller
        return torch.tanh(self.net(torch.cat([obs / 100.0, z], dim=-1)))


# ==============================================================================
# Running mean/std for value-target normalization (PopArt-lite). [CORR-B]
# ------------------------------------------------------------------------------
# Replaces the hand-tuned reward_scale. Parallel-variance (Chan/Welford) update so
# the stats converge over training and stop drifting; we de-normalize the critic
# output for GAE and normalize the regression target so the critic always predicts
# a ~unit-scale value regardless of the demand regime's cost magnitude.
# ==============================================================================
class RunningNorm:
    def __init__(self, eps=1e-4):
        self.mean = 0.0
        self.var = 1.0
        self.count = float(eps)

    def update(self, x):
        x = x.detach().reshape(-1).float()
        n = x.numel()
        if n == 0:
            return
        bmean = float(x.mean().item())
        bvar = float(x.var(unbiased=False).item())
        delta = bmean - self.mean
        tot = self.count + n
        self.mean = self.mean + delta * n / tot
        m_a = self.var * self.count
        m_b = bvar * n
        self.var = (m_a + m_b + delta * delta * self.count * n / tot) / tot
        self.count = tot

    @property
    def std(self):
        return float(max(self.var, 1e-6) ** 0.5)


# ==============================================================================
# Module D -- BELIEF-CONDITIONED, PER-AGENT, SCALAR CRITIC  [CORR-A]
# ------------------------------------------------------------------------------
# forward(state, belief) -> [B, N] (one scalar value per agent, NORMALIZED scale;
# the trainer de-normalizes via RunningNorm). The critic ingests BOTH the global
# state AND the detached per-agent demand belief z (flattened across agents): the
# global state has NO demand-rate channel, so under domain randomization the value
# baseline could not subtract lambda variance without z. Causal encoder => no
# leakage; z detached => critic loss cannot corrupt the encoder; training-only.
#
# (Was a 32-quantile critic whose quantiles all regressed to the SCALAR lambda-
# return -> collapsed -> a mean critic in disguise. The risk path uses empirical
# returns, not the critic, so scalar loses nothing and is faster.)
# ==============================================================================
class ValueCritic(nn.Module):
    def __init__(self, state_dim, belief_dim, hidden, n_agents, n_quantiles=1):
        super().__init__()
        self.n_agents = n_agents
        self.n_q = int(n_quantiles)          # accepted for interface compat; unused (scalar)
        self.belief_dim = belief_dim         # = n_agents * z_dim (flattened)
        self.gdim = state_dim
        self.hidden = hidden
        self.backbone = nn.Sequential(nn.Linear(state_dim + belief_dim, hidden), nn.ReLU(),
                                      nn.Linear(hidden, hidden), nn.ReLU())
        self.heads = nn.Linear(hidden, n_agents)   # N independent scalar value heads

    def _input(self, state, belief):
        # state [B, state_dim]; belief [B, N, z] or [B, N*z] (DETACHED by caller).
        b = belief.reshape(belief.size(0), -1)     # [B, N*z]
        return torch.cat([state / 100.0, b], dim=-1)

    def forward(self, state, belief):
        h = self.backbone(self._input(state, belief))
        return self.heads(h)                       # [B, N]  (normalized scale)

    # kept for API symmetry; trainer calls forward + de-normalizes itself
    def value(self, state, belief):
        return self.forward(state, belief)


# Import-compat alias: train_draco_v4.py imports the name `DistributionalCritic`.
DistributionalCritic = ValueCritic


# ==============================================================================
# Rollout buffer (one per episode). Stores PER-AGENT raw local costs (shaping
# applied in the trainer so beta is a single config knob) and the S-action
# log-prob only (messages carry no log-prob under DIAL).
# ==============================================================================
class DRACORolloutBuffer:
    def __init__(self):
        self.clear()

    def clear(self):
        for k in ("obs", "g", "msg_in", "S_act", "logp", "cost", "done", "demand_tgt"):
            setattr(self, k, [])

    def push(self, **kw):
        for k, v in kw.items():
            getattr(self, k).append(v)

    def __len__(self):
        return len(self.obs)


# ==============================================================================
# DRACOTrainerV4 (REVIEW-CORRECTED) -- MAPPO actor + DIAL messages +
# belief-conditioned SCALAR critic (value-normalized, no target net) +
# self-supervised demand-ELBO encoder.
# ==============================================================================
class DRACOTrainerV4:
    def __init__(self, encoder, actors, msg_heads, critic, cfg, device, n_agents, adj):
        self.encoder, self.actors, self.msg_heads, self.critic = encoder, actors, msg_heads, critic
        self.device = device
        self.N = n_agents
        self.adj = adj.to(device)
        self.msg_mode = cfg.get("msg_mode", "learned")   # "learned" DIAL | "dhat" broadcast (Phase 3)

        self.gamma = float(cfg.get("gamma", 0.99))
        self.gae_lambda = float(cfg.get("gae_lambda", 0.95))
        self.k_epochs = int(cfg.get("k_epochs", 4))
        self.critic_epochs = int(cfg.get("critic_epochs", cfg.get("k_epochs", 4)))
        self.eps_clip = float(cfg.get("eps_clip", 0.2))
        self.max_grad_norm = float(cfg.get("max_grad_norm", 0.5))
        self.entropy_coef = float(cfg.get("entropy_coef", 0.01))
        self.pred_coef = float(cfg.get("pred_coef", 1.0))
        self.kl_coef = float(cfg.get("kl_coef", cfg.get("ib_beta", 1e-3)))
        self.msg_penalty = float(cfg.get("msg_penalty_coef", 0.0))
        self.use_comm = bool(cfg.get("use_comm", True))
        self.use_context = bool(cfg.get("use_context", True))
        self.srdqn_beta = float(cfg.get("srdqn_beta", 0.5))
        self.cvar_alpha = float(cfg.get("cvar_alpha", 0.2))
        self.risk_eta = float(cfg.get("risk_eta", 0.0))
        self.s_smooth_coef = float(cfg.get("s_smooth_coef", 0.0))
        self.demand_aux_coef = float(cfg.get("demand_aux_coef", 0.05))
        self.critic_uses_belief = bool(cfg.get("critic_uses_belief", True))
        self.belief_sample = bool(cfg.get("belief_sample", False))
        # ordering safety valves (OFF by default; see module docstring)
        self.order_cap_coef = float(cfg.get("order_cap_coef", 0.0))
        self.corr_l2_coef = float(cfg.get("corr_l2_coef", 0.0))

        # [CORR-B] value-target normalizer replaces reward_scale (no longer read).
        self.ret_norm = RunningNorm()

        actor_params = [p for a in self.actors for p in a.parameters()]
        msg_params = [p for m in self.msg_heads for p in m.parameters()]
        self.policy_opt = torch.optim.Adam(
            [{"params": actor_params, "lr": cfg.get("lr_actor", 3e-4)},
             {"params": msg_params, "lr": cfg.get("lr_msg", 3e-4)}])
        self.critic_opt = torch.optim.Adam(critic.parameters(), lr=cfg.get("lr_critic", 1e-3))
        self.enc_opt = torch.optim.Adam(encoder.parameters(), lr=cfg.get("lr_encoder", 3e-4))

    # ---- belief read (detached for the policy/critic) ----
    def _encode_belief(self, obs, msg_in):
        m = msg_in if self.use_comm else torch.zeros_like(msg_in)
        mu, logstd, _ = self.encoder.forward_sequence(obs, m)
        if self.belief_sample:
            z = mu + torch.randn_like(mu) * logstd.exp()
        else:
            z = mu
        return z.detach()

    def _zero_belief(self, z):
        # flatten [T,N,z] -> [T, N*z] for the critic; zero it out if disabled
        T = z.size(0)
        zc = z.reshape(T, -1)
        return zc if self.critic_uses_belief else torch.zeros_like(zc)

    # ---- de-normalized critic value for GAE (critic predicts a unit-scale value) ----
    def _critic_value(self, g, zc):
        return self.critic(g, zc) * self.ret_norm.std + self.ret_norm.mean      # [T,N]

    # ---- finite-horizon GAE with 0-bootstrap at truncation (critic is time-aware) ----
    def _gae(self, rew, val, done):
        T = rew.size(0)
        ve = torch.cat([val, torch.zeros(1, device=self.device)], dim=0)
        adv = torch.zeros(T, device=self.device)
        last = torch.zeros((), device=self.device)
        for t in reversed(range(T)):
            nonterm = 1.0 - done[t]
            delta = rew[t] + self.gamma * ve[t + 1] * nonterm - val[t]
            last = delta + self.gamma * self.gae_lambda * nonterm * last
            adv[t] = last
        return adv

    def update(self, episode_buffers):
        dev, N = self.device, self.N
        E = []
        for b in episode_buffers:
            E.append(dict(
                obs=torch.stack(b.obs), g=torch.stack(b.g), msg_in=torch.stack(b.msg_in),
                S_act=torch.stack(b.S_act), old_logp=torch.stack(b.logp),
                cost=torch.stack(b.cost), done=torch.stack(b.done),
                dtgt=torch.stack(b.demand_tgt),
            ))

        # SRDQN per-agent shaped reward in RAW units ([CORR-B]: normalization handles scale).
        for d in E:
            c = d["cost"]
            others = c.sum(dim=-1, keepdim=True) - c
            d["rew"] = -(c + self.srdqn_beta * others)

        # belief + advantages (baseline = ONLINE critic, de-normalized) [CORR-C: no target net]
        with torch.no_grad():
            for d in E:
                d["z"] = self._encode_belief(d["obs"], d["msg_in"])
                d["zc"] = self._zero_belief(d["z"])
                V = self._critic_value(d["g"], d["zc"])                  # [T,N] de-normalized
                done_t = d["done"].squeeze(-1)
                adv = torch.stack([self._gae(d["rew"][:, i], V[:, i], done_t) for i in range(N)], dim=1)
                d["V"], d["adv"] = V, adv
                # [FIX 2 kept] critic regression target = GAE lambda-return = V + adv (real scale)
                d["vtarget"] = (V + adv).detach()
            all_adv = torch.cat([d["adv"] for d in E], dim=0)
            a_mean, a_std = all_adv.mean(dim=0), all_adv.std(dim=0) + 1e-8
            for d in E:
                d["adv"] = (d["adv"] - a_mean) / a_std
            # [CORR-B] refresh the value-target normalizer from the real-scale returns
            self.ret_norm.update(torch.cat([d["vtarget"].reshape(-1) for d in E]))
            if self.risk_eta > 0:
                # trajectory-level CVaR reweight (Tamar-style), EMPIRICAL returns -- not the critic
                team_ret = torch.stack([-(d["cost"].sum()) for d in E])
                k = max(1, int(math.ceil(self.cvar_alpha * len(E))))
                thresh = torch.topk(team_ret, k, largest=False).values.max()
                for j, d in enumerate(E):
                    if (team_ret[j] <= thresh).item():
                        d["adv"] = d["adv"] * (1.0 + self.risk_eta)

        rn_mean, rn_std = self.ret_norm.mean, self.ret_norm.std
        a_loss_tot = c_loss_tot = e_loss_tot = 0.0

        # ---- CRITIC: regress the (normalized) value to the normalized lambda-return ----
        for _ in range(self.critic_epochs):
            self.critic_opt.zero_grad()
            closs = 0.0
            for d in E:
                v_norm = self.critic(d["g"], d["zc"])                              # [T,N]
                target_norm = ((d["vtarget"] - rn_mean) / rn_std).detach()
                loss = F.mse_loss(v_norm, target_norm) / len(E)
                loss.backward(); closs += loss.item()
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
            self.critic_opt.step()
            c_loss_tot += closs

        # ---- ENCODER: demand-reconstruction ELBO. [FIX 3 kept] reconstruct RAW demand ----
        for _ in range(self.k_epochs):
            self.enc_opt.zero_grad()
            eloss = 0.0
            for d in E:
                m_in = d["msg_in"] if self.use_comm else torch.zeros_like(d["msg_in"])
                mu, logstd, pred = self.encoder.forward_sequence(d["obs"], m_in)
                pred_loss = F.mse_loss(pred, d["dtgt"])                              # [FIX 3] no /100
                kl = (-0.5 * (1 + 2 * logstd - mu.pow(2) - (2 * logstd).exp())).mean()
                loss = (self.pred_coef * pred_loss + self.kl_coef * kl) / len(E)
                loss.backward(); eloss += loss.item()
            torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), self.max_grad_norm)
            self.enc_opt.step()
            e_loss_tot += eloss

        # ---- ACTOR + DIAL messages (MAPPO surrogate; cross-agent gradient via ADJ) ----
        for _ in range(self.k_epochs):
            self.policy_opt.zero_grad()
            ploss = 0.0
            for d in E:
                T = d["obs"].size(0)
                z = d["z"]; z_act = z if self.use_context else torch.zeros_like(z)
                if self.use_comm:
                    if self.msg_mode == "dhat":     # broadcast detached d_hat: message is a readout, no DIAL gradient
                        m = torch.stack([self.actors[i].demand_estimate(z[:, i]).detach() for i in range(N)], dim=1)
                    else:
                        m = torch.stack([self.msg_heads[i](d["obs"][:, i], z[:, i]) for i in range(N)], dim=1)
                    routed = torch.einsum("ij,tjm->tim", self.adj, m)
                    incoming = torch.cat([torch.zeros(1, N, m.size(-1), device=dev), routed[:-1]], dim=0)
                else:
                    m = None
                    incoming = torch.zeros(T, N, d["msg_in"].size(-1), device=dev)
                loss = torch.zeros((), device=dev)
                for i in range(N):
                    s_mu, s_std = self.actors[i](d["obs"][:, i], z_act[:, i], incoming[:, i])
                    logp = Normal(s_mu, s_std).log_prob(d["S_act"][:, i]).sum(-1, keepdim=True)
                    ratio = torch.exp(logp - d["old_logp"][:, i])
                    A = d["adv"][:, i].unsqueeze(-1)
                    surr1 = ratio * A
                    surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * A
                    ent = Normal(s_mu, s_std).entropy().mean()
                    loss = loss - torch.min(surr1, surr2).mean() - self.entropy_coef * ent
                    if self.s_smooth_coef > 0 and s_mu.size(0) > 1:
                        loss = loss + self.s_smooth_coef * (s_mu[1:] - s_mu[:-1]).abs().mean()
                    if self.demand_aux_coef > 0 and hasattr(self.actors[i], "demand_estimate"):
                        d_hat = self.actors[i].demand_estimate(z_act[:, i])
                        loss = loss + self.demand_aux_coef * F.mse_loss(d_hat, d["dtgt"][:, i])
                    # --- ordering safety valves (OFF by default) ---
                    if self.order_cap_coef > 0:
                        oi = d["obs"][:, i]
                        IP = oi[..., 0:1] - oi[..., 1:2] + oi[..., 2:3]
                        mo = float(self.actors[i].max_order)
                        over = F.relu((s_mu - IP - mo) / mo)
                        loss = loss + self.order_cap_coef * over.mean()
                    if self.corr_l2_coef > 0 and getattr(self.actors[i], "_last_corr", None) is not None:
                        loss = loss + self.corr_l2_coef * self.actors[i]._last_corr.pow(2).mean()
                if self.use_comm and self.msg_penalty > 0:
                    loss = loss + self.msg_penalty * m.pow(2).mean()
                loss = loss / len(E)
                loss.backward(); ploss += loss.item()
            params = [p for a in self.actors for p in a.parameters()] + \
                     [p for mh in self.msg_heads for p in mh.parameters()]
            torch.nn.utils.clip_grad_norm_(params, self.max_grad_norm)
            self.policy_opt.step()
            a_loss_tot += ploss

        kk = max(1, self.k_epochs)
        return a_loss_tot / kk, c_loss_tot / max(1, self.critic_epochs), e_loss_tot / kk