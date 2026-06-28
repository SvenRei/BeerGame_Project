"""
baselines.py  --  Self-contained base-stock + behavioral baselines for the DRACO
beer-game thesis. No dependency on the analysis scripts.

POLICIES you run ON the env (editable like the agents):

  OptimizedBaseStockPolicy : per-echelon installation base-stock,
                          order_i = clip(S_i - IP_i, 0, max_order).
                      Get the LEVELS from optimize_basestock_on_env(...) (env-matched) or
                      coord_descent_serial(...) (idealized anchor). For a serial system an
                      echelon base-stock attains the optimal cost (Clark & Scarf 1960;
                      installation/echelon cost-equivalence, Axsater & Rosling 1993), but
                      these LEVELS are numerically optimized, NOT the analytic echelon DP --
                      so do not call them "Clark-Scarf optimal" unless you run the canonical
                      DP (see `canonical`). Back-compat alias: ClarkScarfPolicy.

  AdaptiveForecastPolicy   : exponentially-smoothed demand mean+variance -> order-up-to at a
                      normal-approx critical fractile. The "naive practitioner forecast" rung;
                      its EMPIRICAL variance captures bullwhip inflation upstream, so it is a
                      strong (not strawman) non-RL adaptive baseline.

  BayesPoissonPolicy       : per-stage Gamma-Poisson conjugate belief on the demand rate ->
                      base-stock at the critical fractile of the negative-binomial predictive
                      over the protection interval. Bayes-OPTIMAL at the retailer (Poisson
                      demand; Scarf 1959, Azoury 1985); a HEURISTIC upstream (whose incoming
                      ORDER stream is not Poisson).
                      FRAMING (important -- read before reporting): this is the single-stage-
                      optimal adaptive policy, but in the multi-echelon penalty-at-EVERY-stage
                      cost it BULLWHIPS (changing its order-up-to level injects order variance)
                      and is typically WORSE than the static BAR. So report it as a NAIVE-
                      ADAPTATION FLOOR, NOT "the bar to beat." The headline comparators are the
                      static BAR (bullwhip-free, the hard target) and the per-lambda Oracle;
                      DRACO-vs-Bayes is the SECONDARY "beats naive adaptation" check. (Under the
                      CANONICAL retailer-only cost the textbook ordering Bayes < BAR is more
                      likely to hold.) Use make_bayes_rung() (retailer-Bayes + adaptive upstream)
                      for the reported rung.
                      SPEC (what the policy knows): Gamma prior centred at the TRAINING-support
                      mean (prior_mean); updates the rate from each period's observed demand;
                      critical-fractile order-up-to over protection interval tau; finite-horizon
                      with no terminal correction; retailer sees true customer demand, upstream
                      stages see only their own (amplified) incoming order stream.

  StermanPolicy            : Sterman (1989) anchoring-and-adjustment heuristic. The behavioral
                      bullwhip baseline / floor.

THEORY ANCHORS (no env, pure numpy):
  the idealized serial model + base-stock optimizer (selftest validates convexity), and the
  CANONICAL serial model (penalty at the retailer only) whose cost is provably convex
  (Clark-Scarf 1960) so the optimizer is provably global -- the verified CEILING anchor.

C1 REFERENCES (the headline regime-uncertainty ladder) -- use `regime`:
  regime_benchmark() computes a FOUR-rung ladder
      static BAR  <  Adaptive  <=  Bayes  <=  per-lambda Oracle
  with DISJOINT select/eval seeds (no winner's-curse): levels are SELECTED on SELECT_SEED_BASE
  and every rung is REPORTED on EVAL_SEED_BASE (== the DRACO held-out seeds, so CRN-comparable).
  Writes results/baselines_regime_v2.json (read by c1_stats.py and the trainer/eval ref loader).

PROTECTION INTERVAL tau (per stage, for the Bayes/Adaptive rungs) = effective_lead + 1 review,
read from the env's lead structure: order(2)+ship(2)=4 downstream, order_mfr(1)+production(2)=3
mfr -> DEFAULT_TAU=[5,5,5,4]. UPDATE this if you change lead times, or the rungs miscalibrate.

Run:
  python scripts/baselines.py selftest
  python scripts/baselines.py regime    --select-episodes 80 --eval-episodes 200
  python scripts/baselines.py canonical --lambdas 6 10 14 18 22         # provable optimum anchor
  python scripts/baselines.py serial    --lam 8 --L 4 --h 0.5 --b 1.0
  python scripts/baselines.py optimize  --regimes poisson black_swan extreme_chaos
  python scripts/baselines.py eval      --regimes poisson --seeds 20 --cs-levels 44 56 64 68
"""
import argparse
import os
import sys
import json
import numpy as np
from collections import deque
from scipy import stats

AGENTS = ["retailer", "wholesaler", "distributor", "manufacturer"]

# Default env settings (mirror conf/config.yaml `env:`). EDIT here or via the env config.
ENV_BASE = {"horizon": 50, "max_order": 100, "holding_cost": 0.5, "backorder_cost": 1.0}
SEED_BASE = 100000          # held-out eval seed space (== the DRACO held-out eval seeds)

# Disjoint seed blocks (fixes in-sample optimism): SELECT tunes baseline levels; EVAL reports
# every rung. EVAL == SEED_BASE so baselines and the DRACO held-out eval (also SEED_BASE+e) are
# scored on the SAME episodes (CRN), while baseline SELECTION uses a block DRACO never touches.
SELECT_SEED_BASE = SEED_BASE + 500000
EVAL_SEED_BASE = SEED_BASE

# Held-out demand regimes for the C1 study. Each lambda is a STATIONARY poisson rate the policy
# does not know in advance. Keep in sync with HELDOUT_LAMBDAS in the trainer's held-out eval.
HELDOUT_LAMBDAS = [6.0, 10.0, 14.0, 18.0, 22.0]

# Protection interval per echelon for the Bayes/Adaptive rungs (effective_lead + 1 review).
DEFAULT_TAU = [5.0, 5.0, 5.0, 4.0]


# ==============================================================================
# Env access + rollout harness (the ONE place policies meet the environment)
# ==============================================================================
def get_env_class():
    """Import the project env. Run from the repo root (script lives in scripts/)."""
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from envs.beer_game_env import BeerGameParallelEnv
    return BeerGameParallelEnv


def _ip(o):
    """Inventory position from an obs [inventory, backlog, on_order, last_incoming]."""
    return float(o[0]) - float(o[1]) + float(o[2])


def _action(order_units, max_order):
    """Env action = a single fraction in [0,1]; env reconstructs round(frac*max_order)."""
    return [float(np.clip(order_units / max_order, 0.0, 1.0))]


def rollout(env, policy, seed):
    """One deterministic episode of `policy` on `env`. Returns (total_cost, per_agent_cost).
    Cost is read from the env's info dict, exactly as the trainer/agent sees it."""
    obs, _ = env.reset(seed=seed)
    policy.reset()
    total = 0.0
    per_agent = {a: 0.0 for a in AGENTS}
    while env.agents:
        acts = policy.act(obs, env)
        obs, _, _, trunc, info = env.step(acts)
        total += float(info[AGENTS[0]]["supply_chain_cost"])
        for a in AGENTS:
            per_agent[a] += float(info[a]["local_cost"])
        if any(trunc.values()):
            break
    return total, per_agent


# ==============================================================================
# POLICIES  (editable like the agents: configure via the constructor)
# ==============================================================================
class Policy:
    """Base policy. act(obs_dict, env) -> {agent: [fraction in 0..1]}.
    reset() clears any per-episode internal state (called at the start of each episode)."""
    def reset(self):
        pass

    def act(self, obs, env):
        raise NotImplementedError


def _per_echelon(x, name):
    """Accept a scalar (shared across echelons) or a length-4 list (per echelon)."""
    if np.isscalar(x):
        return [float(x)] * 4
    x = [float(v) for v in x]
    if len(x) != 4:
        raise ValueError(f"{name} must be a scalar or length-4 list, got {x}")
    return x


def _critical_fractile(h, b):
    return b / (b + h)


class BaseStockPolicy(Policy):
    """Per-echelon installation base-stock: order_i = clip(S_i - IP_i, 0, max_order).
    IP_i = inventory - backlog + on_order  (recoverable from the 4-scalar obs).

    EDIT: S = scalar (same level everywhere) or [S_ret, S_whole, S_dist, S_mfr]."""
    def __init__(self, S):
        self.S = _per_echelon(S, "S")

    def act(self, obs, env):
        return {a: _action(max(0.0, self.S[i] - _ip(obs[a])), env.max_order)
                for i, a in enumerate(AGENTS)}


class OptimizedBaseStockPolicy(BaseStockPolicy):
    """Numerically-optimized per-echelon installation base-stock. Identical *mechanism* to
    BaseStockPolicy; the point is the LEVELS.

    For a serial system an echelon base-stock attains the optimal cost (Clark & Scarf 1960),
    and installation vs echelon are cost-equivalent at the optimum (Axsater & Rosling 1993) --
    BUT these levels are found by coordinate descent on the env / idealized sim, NOT by the
    analytic echelon DP. Do NOT label them "Clark-Scarf optimal" unless you run the canonical
    DP (`canonical`), which is provably global for the canonical cost structure.

    Get the levels by either:
      * optimize_basestock_on_env(regime, ...)   <- env-matched (the number you report), or
      * coord_descent_serial(lam, L=4, ...)       <- idealized theory anchor,
    then construct OptimizedBaseStockPolicy(levels)."""
    pass


# Back-compat: old call sites / eval scripts that import ClarkScarfPolicy keep working.
ClarkScarfPolicy = OptimizedBaseStockPolicy


class AdaptiveForecastPolicy(Policy):
    """Exponentially-weighted demand mean & variance per stage; order-up-to at a normal-approx
    critical fractile. The empirical variance lets the level inflate where the order stream is
    noisier (bullwhip), so this is a strong non-RL adaptive rung.
    EDIT: tau (protection interval per echelon), eta (smoothing)."""
    def __init__(self, tau=None, h=0.5, b=1.0, eta=0.2):
        self.tau = _per_echelon(DEFAULT_TAU if tau is None else tau, "tau")
        self.z = float(stats.norm.ppf(_critical_fractile(h, b)))
        self.eta = float(eta)
        self.mu = self.var = None

    def reset(self):
        self.mu = [8.0] * 4
        self.var = [4.0] * 4

    def act(self, obs, env):
        acts = {}
        for i, a in enumerate(AGENTS):
            o = obs[a]
            diff = float(o[3]) - self.mu[i]
            self.mu[i] += self.eta * diff
            self.var[i] = (1.0 - self.eta) * (self.var[i] + self.eta * diff * diff)
            lead_mu = self.mu[i] * self.tau[i]
            lead_sd = np.sqrt(max(1e-6, self.var[i] * self.tau[i]))
            S = max(0.0, lead_mu + self.z * lead_sd)
            acts[a] = _action(max(0.0, S - _ip(o)), env.max_order)
        return acts


class BayesPoissonPolicy(Policy):
    """Per-stage Gamma-Poisson conjugate belief on the demand rate. Posterior after t periods:
    Gamma(alpha0 + sum d, beta0 + t). Demand over the protection interval tau marginalizes to
    NegBin(r=alpha, p=beta/(beta+tau)); base-stock = its critical-fractile quantile.

    Bayes-OPTIMAL at the retailer (Poisson customer demand; Scarf 1959, Azoury 1985); a strong
    HEURISTIC at upstream stages (whose incoming order stream is not Poisson). Prior defaults to
    the TRAINING regime support so DRACO and this rung share the same prior over lambda.
    EDIT: tau, prior_mean, prior_strength (pseudo-periods)."""
    def __init__(self, tau=None, h=0.5, b=1.0, prior_mean=14.0, prior_strength=0.3):
        self.tau = _per_echelon(DEFAULT_TAU if tau is None else tau, "tau")
        self.frac = _critical_fractile(h, b)
        self.a0 = float(prior_mean) * float(prior_strength)
        self.b0 = float(prior_strength)
        self.alpha = self.beta = None

    def reset(self):
        self.alpha = [self.a0] * 4
        self.beta = [self.b0] * 4

    def act(self, obs, env):
        acts = {}
        for i, a in enumerate(AGENTS):
            o = obs[a]
            self.alpha[i] += max(0.0, float(o[3]))
            self.beta[i] += 1.0
            p = self.beta[i] / (self.beta[i] + self.tau[i])
            S = float(stats.nbinom.ppf(self.frac, self.alpha[i], p))
            acts[a] = _action(max(0.0, S - _ip(o)), env.max_order)
        return acts


def ar1_cumulative_forecast(d_t, mu, rho, sigma, tau):
    """Minimum-variance forecast of CUMULATIVE demand over the next tau periods for an AR(1)
    process d_t = mu + rho*(d_{t-1}-mu) + eps,  eps ~ N(0, sigma^2). Returns (mean, sd).

      mean = sum_{k=1..tau} E[d_{t+k}|d_t] = tau*mu + (d_t-mu) * rho*(1-rho^tau)/(1-rho)
      sd   : the cumulative forecast error regroups to
             sum_{j=1..tau} eps_{t+j} * (1-rho^{tau-j+1})/(1-rho), so
             Var = sigma^2 * sum_{j=1..tau} [(1-rho^{tau-j+1})/(1-rho)]^2
      rho -> 0 recovers the i.i.d. case (mean=tau*mu, sd=sigma*sqrt(tau))."""
    tau = int(round(tau))
    if tau <= 0:
        return 0.0, 0.0
    if abs(rho) < 1e-9:
        return tau * float(mu), float(sigma) * np.sqrt(tau)
    geom = rho * (1.0 - rho ** tau) / (1.0 - rho)
    mean = tau * mu + (d_t - mu) * geom
    var = sum(((1.0 - rho ** (tau - j + 1)) / (1.0 - rho)) ** 2 for j in range(1, tau + 1)) * sigma ** 2
    return float(mean), float(np.sqrt(max(0.0, var)))


class AR1BaseStockPolicy(Policy):
    """MMFE / Kalman-optimal base-stock for AR(1) demand -- the PROPER comparator for the AR(1)
    family. The Gamma-Poisson BayesPoissonPolicy is NOT optimal under autocorrelation, so on AR(1)
    you must compare DRACO to THIS, not to Bayes. Sets the order-up-to level at the critical-fractile
    quantile of the closed-form cumulative-demand forecast over the protection interval:
        S = forecast_mean_over_tau + z * forecast_sd_over_tau,   z = Phi^-1(b/(b+h)).

    Optimal at the RETAILER (sees the true AR(1) customer demand); a strong heuristic upstream (whose
    incoming ORDER stream is a transformed process) -- exactly the same retailer-optimal / upstream-
    heuristic status as the Bayes rung. Pass the env's AR(1) params (mu, rho, sigma): this is the
    *informed* optimum, parallel to Bayes being informed about the Poisson family.
    Heath & Jackson (1994, MMFE); Graves (1999); Lee, So & Tang (2000)."""
    def __init__(self, mu=12.0, rho=0.6, sigma=3.0, tau=None, h=0.5, b=1.0):
        self.mu = float(mu)
        self.rho = float(rho)
        self.sigma = float(sigma)
        self.tau = _per_echelon(DEFAULT_TAU if tau is None else tau, "tau")
        self.z = float(stats.norm.ppf(_critical_fractile(h, b)))

    def reset(self):
        pass                              # memoryless given d_t: the AR(1) forecast needs only the last demand

    def act(self, obs, env):
        acts = {}
        for i, a in enumerate(AGENTS):
            o = obs[a]
            d = float(o[3])               # last realized incoming demand/order = the AR(1) state d_t
            mean, sd = ar1_cumulative_forecast(d, self.mu, self.rho, self.sigma, self.tau[i])
            S = max(0.0, mean + self.z * sd)
            acts[a] = _action(max(0.0, S - _ip(o)), env.max_order)
        return acts


class RetailerOptimalPolicy(Policy):
    """The DEFENSIBLE multi-echelon adaptive comparator: the family-OPTIMAL model at the retailer
    (where it is valid -- the retailer faces the true customer demand), and a robust, variance-aware
    ADAPTIVE forecast base-stock at the upstream echelons (where the incoming ORDER stream is
    bullwhipped and NOT the assumed family).

    WHY this exists: applying a Gamma-Poisson (or AR(1)) conjugate at EVERY stage assumes the order
    stream upstream is Poisson/AR(1). It is not -- it is the amplified order process -- so the naive
    rung inflates ABOVE a static base-stock (Bayes > BAR), which is nonsensical for a 'near-optimal
    adaptive' policy and makes 'DRACO beats Bayes' meaningless. Composing retailer-optimal with an
    EWMA critical-fractile upstream (which estimates the order stream's mean AND variance, so it does
    not misfire on the bullwhipped stream) restores the sane ordering Oracle <= rung <= BAR.

    Retailer optimality: Scarf (1959) / Azoury (1985) for Poisson (and NegBin = its Gamma-Poisson
    predictive); Heath & Jackson (1994) for AR(1). Upstream: AdaptiveForecastPolicy."""
    def __init__(self, retailer_policy, upstream_policy):
        self.retailer = retailer_policy
        self.upstream = upstream_policy

    def reset(self):
        self.retailer.reset()
        self.upstream.reset()

    def act(self, obs, env):
        r = self.retailer.act(obs, env)
        u = self.upstream.act(obs, env)
        return {a: (r[a] if a == AGENTS[0] else u[a]) for a in AGENTS}


def make_bayes_rung(tau=None, h=0.5, b=1.0, prior_mean=14.0):
    """The corrected Bayes-adaptive rung: Gamma-Poisson conjugate at the retailer, adaptive EWMA
    upstream (see RetailerOptimalPolicy). Use this everywhere instead of a bare BayesPoissonPolicy."""
    return RetailerOptimalPolicy(BayesPoissonPolicy(tau=tau, h=h, b=b, prior_mean=prior_mean),
                                 AdaptiveForecastPolicy(tau=tau, h=h, b=b))


def make_ar1_rung(mu=12.0, rho=0.6, sigma=3.0, tau=None, h=0.5, b=1.0):
    """The corrected AR(1) rung: MMFE/AR(1)-optimal at the retailer, adaptive EWMA upstream."""
    return RetailerOptimalPolicy(AR1BaseStockPolicy(mu=mu, rho=rho, sigma=sigma, tau=tau, h=h, b=b),
                                 AdaptiveForecastPolicy(tau=tau, h=h, b=b))


class StermanPolicy(Policy):
    """Sterman (1989) anchoring-and-adjustment ordering heuristic -- the behavioral bullwhip
    baseline.

        Lhat_t = theta * Lhat_{t-1} + (1 - theta) * incoming_t      (demand anchor)
        O_t    = max(0, Lhat_t + alpha_S * (S_prime - S_eff - beta * SL))

      incoming_t : last realized incoming order/demand  (obs[3])
      S_eff      : effective inventory = on_hand - backlog  (obs[0] - obs[1])
      SL         : supply line = on-order                  (obs[2])
      beta = 1 -> rational (non-bullwhip); beta < 1 -> supply-line under-weighting -> bullwhip.
    EDIT any parameter as scalar (shared) or length-4 (per echelon)."""
    def __init__(self, theta=0.25, alpha_S=0.36, beta=0.34, S_prime=17.0, L_init=8.0):
        self.theta = _per_echelon(theta, "theta")
        self.alpha_S = _per_echelon(alpha_S, "alpha_S")
        self.beta = _per_echelon(beta, "beta")
        self.S_prime = _per_echelon(S_prime, "S_prime")
        self.L_init = _per_echelon(L_init, "L_init")
        self.Lhat = None

    def reset(self):
        self.Lhat = list(self.L_init)

    def act(self, obs, env):
        acts = {}
        for i, a in enumerate(AGENTS):
            o = obs[a]
            incoming = float(o[3])
            self.Lhat[i] = self.theta[i] * self.Lhat[i] + (1.0 - self.theta[i]) * incoming
            s_eff = float(o[0]) - float(o[1])
            supply_line = float(o[2])
            order = self.Lhat[i] + self.alpha_S[i] * (self.S_prime[i] - s_eff - self.beta[i] * supply_line)
            acts[a] = _action(max(0.0, order), env.max_order)
        return acts


# ==============================================================================
# Evaluate any set of policies ON THE ENV, across regimes x seeds (common seeds = CRN)
# ==============================================================================
def evaluate(named_policies, regimes, n_seeds=20, seed_base=SEED_BASE,
             env_cfg=None, env_class=None, verbose=True):
    """named_policies: dict name -> Policy. Returns {name: {regime: (mean, std, per_agent_mean)}}."""
    env_cfg = dict(ENV_BASE if env_cfg is None else env_cfg)
    Env = env_class or get_env_class()
    out = {name: {} for name in named_policies}
    for reg in regimes:
        for name, pol in named_policies.items():
            costs, pa = [], {a: [] for a in AGENTS}
            for s in range(n_seeds):
                env = Env({**env_cfg, "demand_type": reg})
                c, per = rollout(env, pol, seed_base + s)
                costs.append(c)
                for a in AGENTS:
                    pa[a].append(per[a])
            out[name][reg] = (float(np.mean(costs)), float(np.std(costs)),
                              {a: float(np.mean(pa[a])) for a in AGENTS})
    if verbose:
        _print_eval_table(out, regimes, n_seeds)
    return out


def _print_eval_table(out, regimes, n_seeds):
    names = list(out.keys())
    print("=" * (24 + 26 * len(names)))
    print(f"POLICY EVALUATION ON ENV  |  {n_seeds} seeds/regime (common random numbers)")
    print("=" * (24 + 26 * len(names)))
    head = f"{'regime':<16}" + "".join(f"{n:>26}" for n in names)
    print(head)
    print("-" * len(head))
    for reg in regimes:
        row = f"{reg:<16}"
        for n in names:
            m, sd, _ = out[n][reg]
            row += f"{m:>15.1f} +/-{sd:>7.1f}"
        print(row)


# ==============================================================================
# Optimize per-echelon base-stock DIRECTLY ON THE ENV -> OptimizedBaseStock levels.
# (Used by the `optimize`/`eval` CLI for the OOD regimes. The C1 references use the
#  seed-split `regime_benchmark` below, which is authoritative.)
# ==============================================================================
def optimize_basestock_on_env(regime, env_cfg=None, env_class=None, episodes=40,
                              seed_base=SEED_BASE, lo=20, hi=140, step=4, rounds=3, S0=None,
                              verbose=True):
    env_cfg = dict(ENV_BASE if env_cfg is None else env_cfg)
    Env = env_class or get_env_class()

    def cost_of(S_vec):
        pol = BaseStockPolicy(S_vec)
        tot = 0.0
        for e in range(episodes):
            env = Env({**env_cfg, "demand_type": regime})
            c, _ = rollout(env, pol, seed_base + e)
            tot += c
        return tot / episodes

    # Grid-seeded start (replaces the old fixed [44,44,44,44], which biased toward the
    # in-distribution basin -- M3 optimizer fragility).
    if S0 is None:
        gi, gbest = None, np.inf
        for s in range(lo, hi + 1, step):
            c = cost_of([float(s)] * 4)
            if c < gbest:
                gi, gbest = [float(s)] * 4, c
        S, best = gi, gbest
    else:
        S = list(S0); best = cost_of(S)
    for _ in range(rounds):
        improved = False
        for i in range(4):
            for s in range(lo, hi + 1, step):
                cand = list(S); cand[i] = float(s)
                c = cost_of(cand)
                if c < best - 1e-6:
                    S, best, improved = cand, c, True
        if not improved:
            break
    if verbose:
        print(f"  {regime:<14} S*=[{','.join(f'{x:.0f}' for x in S)}]  cost={best:.1f}  "
              f"({episodes} eps/eval)")
    return S, best


# ==============================================================================
# THEORY ANCHOR: idealized serial model + base-stock optimizer (pure numpy, no env)
# ------------------------------------------------------------------------------
# Standard serial beer-game, installation base-stock. stages 0..N-1, 0=retailer. Single
# shipping lead L (pass L=4 to mimic the env's order(2)+ship(2)). penalty_at_retailer_only
# selects the CANONICAL Clark-Scarf cost (holding everywhere, backorder penalty ONLY at the
# retailer) -- for which the cost is provably convex (Clark-Scarf 1960). The default
# (penalty everywhere) matches the project env's service-at-every-stage cost.
# ==============================================================================
def simulate_serial(S, demand, L=2, h=0.5, b=1.0, penalty_at_retailer_only=False):
    S = np.asarray(S, float)
    N = len(S)
    on_hand = S.copy()
    backlog = np.zeros(N)
    on_order = np.zeros(N)
    pipe = [deque([0.0] * L) for _ in range(N)]
    cost = 0.0
    for d in demand:
        for i in range(N):                       # 1. receive
            arr = pipe[i].popleft()
            on_hand[i] += arr
            on_order[i] -= arr
        order = np.zeros(N)                       # 2. order (installation base-stock)
        for i in range(N):
            ip = on_hand[i] - backlog[i] + on_order[i]
            order[i] = max(0.0, S[i] - ip)
            on_order[i] += order[i]
        pipe[N - 1].append(order[N - 1])          # ample external source for the top stage
        incoming = np.zeros(N)                     # 3. ship downstream
        incoming[0] = d
        for i in range(1, N):
            incoming[i] = order[i - 1]
        for i in range(N):
            need = incoming[i] + backlog[i]
            shipped = min(on_hand[i], need)
            on_hand[i] -= shipped
            backlog[i] = need - shipped
            if i > 0:
                pipe[i - 1].append(shipped)
        if penalty_at_retailer_only:              # 4. cost
            cost += float(h * np.sum(on_hand) + b * backlog[0])
        else:
            cost += float(np.sum(h * on_hand + b * backlog))
    return cost


def _demand_samples(lam, T, n_eps, seed):
    rng = np.random.default_rng(seed)
    return rng.poisson(lam, size=(n_eps, T))


def serial_cost(S, lam, L, h, b, T=50, n_eps=200, seed=0, penalty_at_retailer_only=False):
    D = _demand_samples(lam, T, n_eps, seed)
    return float(np.mean([simulate_serial(S, D[e], L, h, b, penalty_at_retailer_only)
                          for e in range(n_eps)]))


def grid_optimize_serial(lam, L, h, b, T=50, n_eps=120, seed=0, lo=0, hi=160, step=4,
                         penalty_at_retailer_only=False):
    best, bestc, N = None, np.inf, 4
    for s in range(lo, hi + 1, step):
        c = serial_cost([s] * N, lam, L, h, b, T, n_eps, seed, penalty_at_retailer_only)
        if c < bestc:
            best, bestc = [float(s)] * N, c
    return best, bestc


def coord_descent_serial(lam, L, h, b, T=50, n_eps=120, seed=0,
                         lo=0, hi=180, step=2, rounds=4, S0=None,
                         penalty_at_retailer_only=False):
    N = 4
    if S0 is None:
        S0 = grid_optimize_serial(lam, L, h, b, T, n_eps, seed, lo, min(hi, 160), step * 2,
                                  penalty_at_retailer_only)[0]
    S = list(S0)
    bestc = serial_cost(S, lam, L, h, b, T, n_eps, seed, penalty_at_retailer_only)
    for _ in range(rounds):
        improved = False
        for i in range(N):
            for s in range(lo, hi + 1, step):
                cand = list(S); cand[i] = float(s)
                c = serial_cost(cand, lam, L, h, b, T, n_eps, seed, penalty_at_retailer_only)
                if c < bestc - 1e-9:
                    S, bestc, improved = cand, c, True
        if not improved:
            break
    return S, bestc


def optimize_canonical_serial(lam, L=4, h=0.5, b=1.0, n_eps=300, seed=0, n_starts=3, verbose=True):
    """Provable global optimum of the CANONICAL serial model (penalty at retailer only): its
    cost is convex (Clark-Scarf 1960), so multi-start coordinate descent finds the global
    optimum. The verified CEILING anchor for the canonical-cost env variant (M3 gold standard).
    Returns (levels, cost)."""
    guess = float(np.clip(lam * (L + 1), 0, 200))
    starts = [[guess] * 4, [0.0] * 4, [200.0] * 4][:max(1, n_starts)]
    best_S, best_c = None, np.inf
    for S0 in starts:
        S, c = coord_descent_serial(lam, L, h, b, n_eps=n_eps, seed=seed, S0=S0,
                                    penalty_at_retailer_only=True)
        if c < best_c:
            best_S, best_c = S, c
    if verbose:
        print(f"  CANONICAL optimum lam={lam:g} L={L} h={h} b={b}: "
              f"S*=[{','.join(f'{x:.0f}' for x in best_S)}]  cost={best_c:.1f}")
    return best_S, best_c


def validate_canonical(lambdas=None, L=4, select_episodes=60, eval_episodes=120, env_cfg=None,
                       env_class=None, lo=0, hi=160, step=8, per_echelon=True, verbose=True):
    """M3.4/M3.5 validation -- turn the env oracle into a VALIDATED COMPUTATIONAL CEILING (not a
    mathematical proof for the implemented env).

    Runs the ENV under the canonical cost (penalty_at_retailer_only=True) and shows:
      (1) the env canonical cost is UNIMODAL in a uniform base-stock (single global basin) -> the
          numerically-optimized env oracle is the global optimum of the base-stock class WITHIN this
          search. Clark-Scarf (1960) establishes that base-stock is the optimal policy CLASS for the
          canonical serial model; the EXACT optimum is verified only for the single stage by
          scripts/dp_optimum.py (the multi-echelon finite-horizon env here is validated, not proven).
      (2) the env oracle's cost MATCHES the idealized convex single-installation model
          (optimize_canonical_serial) in magnitude -> an independent cross-check (differences = the
          env's extra manufacturer production lead + the equilibrium-init transient).

    After this, the defensible claim is: "DRACO recovers X% of the gap to the validated-optimal
    base-stock ceiling (Clark-Scarf 1960 policy class; exact single-stage check via dp_optimum.py)."
    Returns the per-lambda rows."""
    lambdas = [float(l) for l in (HELDOUT_LAMBDAS if lambdas is None else lambdas)]
    base_cfg = dict(ENV_BASE if env_cfg is None else env_cfg)
    cfg_canon = {**base_cfg, "penalty_at_retailer_only": True}          # << canonical cost ON the env
    h, b = base_cfg["holding_cost"], base_cfg["backorder_cost"]
    LamEnv = _make_lambda_env_class(env_class or get_env_class())
    conv_eps = max(40, eval_episodes // 2)

    if verbose:
        print("=" * 90)
        print(f"CANONICAL-OPTIMUM VALIDATION (env penalty_at_retailer_only=True) | lambda "
              f"{[f'{l:g}' for l in lambdas]}")
        print(f"  idealized L={L} (= env order_lead 2 + ship_lead 2). env oracle: "
              f"{'per-echelon' if per_echelon else 'uniform'}, select={select_episodes} eval={eval_episodes} eps.")
        print("=" * 90)
        print(f"  {'lam':>4}{'S_ideal':>17}{'c_ideal':>10}{'S_env(canon)':>22}{'c_env':>10}"
              f"{'1-basin':>9}{'env/ideal':>10}")

    rows, all_unimodal = [], True
    for lam in lambdas:
        S_ideal, c_ideal = optimize_canonical_serial(lam, L=L, h=h, b=b, n_eps=eval_episodes,
                                                     seed=0, verbose=False)
        S_env, _per, _m = best_single_fixed_S([lam], select_episodes, cfg_canon, env_class,
                                              SELECT_SEED_BASE, lo=20, hi=hi, step=4,
                                              per_echelon=per_echelon, verbose=False)
        c_env = _mean_cost_at_lambda(BaseStockPolicy(S_env), lam, eval_episodes, LamEnv,
                                     cfg_canon, EVAL_SEED_BASE)
        # UNIMODALITY of the env canonical cost in a uniform base-stock (noise-tolerant proxy for
        # the Clark-Scarf convexity): decreasing to the min, increasing after -> one global basin.
        grid = list(range(lo, hi + 1, step))
        curve = [_mean_cost_at_lambda(BaseStockPolicy([float(s)] * 4), lam, conv_eps, LamEnv,
                                      cfg_canon, EVAL_SEED_BASE) for s in grid]
        imin = int(np.argmin(curve)); tol = 0.02 * max(curve)
        left = all(curve[i] >= curve[i + 1] - tol for i in range(imin))
        right = all(curve[i] <= curve[i + 1] + tol for i in range(imin, len(curve) - 1))
        unimodal = bool(left and right)
        all_unimodal = all_unimodal and unimodal
        ratio = (c_env / c_ideal) if c_ideal > 0 else float("nan")
        rows.append(dict(lam=lam, S_ideal=S_ideal, c_ideal=c_ideal, S_env=S_env, c_env=c_env,
                         unimodal=unimodal, ratio=ratio, curve=curve, grid=grid))
        if verbose:
            print(f"  {lam:>4g}{str([round(x) for x in S_ideal]):>17}{c_ideal:>10.1f}"
                  f"{str([round(x) for x in S_env]):>22}{c_env:>10.1f}{('yes' if unimodal else 'NO'):>9}"
                  f"{ratio:>9.2f}x")
    if verbose:
        print("-" * 90)
        print(f"  single global basin at every lambda: {all_unimodal}  -> the env canonical oracle "
              f"is the global optimum of the base-stock class within this search.")
        print(f"  Clark-Scarf (1960): base-stock is the optimal policy CLASS for the canonical serial "
              f"model. This is a VALIDATED computational ceiling, not a proof for the finite-horizon")
        print(f"  multi-echelon env (exact optimum proven only for the single stage: scripts/dp_optimum.py).")
        print(f"  env/ideal ~1 cross-checks the magnitude (gap = mfr production lead + finite-horizon "
              f"equilibrium-init transient).")
    return rows


def selftest():
    """Validate the serial optimizer: CD beats-or-matches grid; optimal levels rise with
    lambda and with b/h; cost is convex in a single stage."""
    print("=" * 78)
    print("SELF-TEST: idealized serial base-stock optimizer")
    print("=" * 78)
    L, h, b = 2, 0.5, 1.0
    for lam in (4, 8, 16):
        g, gc = grid_optimize_serial(lam, L, h, b, seed=1)
        s, sc = coord_descent_serial(lam, L, h, b, seed=1, S0=g)
        ok = sc <= gc + 1e-6
        print(f"  lam={lam:2d} | grid S={g[0]:.0f} cost={gc:8.1f} | "
              f"CD S=[{','.join(f'{x:.0f}' for x in s)}] cost={sc:8.1f} | CD<=grid: {ok}")
    s_lo, _ = coord_descent_serial(8, L, 0.5, 0.5, seed=1)
    s_hi, _ = coord_descent_serial(8, L, 0.5, 4.0, seed=1)
    print(f"  b/h sensitivity @lam=8: mean S(b/h=1)={np.mean(s_lo):.1f} "
          f"< mean S(b/h=8)={np.mean(s_hi):.1f} : {np.mean(s_lo) < np.mean(s_hi)}")
    base = list(coord_descent_serial(8, L, h, b, seed=1)[0])
    xs = list(range(10, 60, 5)); cs = []
    for x in xs:
        c = list(base); c[0] = x
        cs.append(serial_cost(c, 8, L, h, b, seed=1))
    convex = np.all(np.diff(np.diff(cs)) >= -1e-6)
    print(f"  convex in S_0 (2nd diff >= 0): {convex}  | curve min at S_0~{xs[int(np.argmin(cs))]}")
    print("  PASS\n" if convex else "  CHECK convexity\n")


# ==============================================================================
# REGIME-UNCERTAINTY env helpers (CRN-comparable to the DRACO held-out eval). A thin subclass
# reads the poisson rate from config 'poisson_lam' -- equivalent to the agent's
# DemandRandomizedBeerGame(lo=hi=lam, p_shift=0): both draw np_random.poisson(lam) off an env
# RNG seeded by `seed`, so these numbers match the DRACO held-out eval at the SAME seeds.
# ==============================================================================
def _make_lambda_env_class(base_cls):
    """Subclass `base_cls` so the poisson rate is read from config key 'poisson_lam'."""
    class _PoissonLambdaEnv(base_cls):
        def _roll_stochastic_demand(self, step):
            if self._config.get("demand_type") == "poisson":
                return self.np_random.poisson(float(self._config.get("poisson_lam", 8.0)))
            return super()._roll_stochastic_demand(step)
    return _PoissonLambdaEnv


def _mean_cost_at_lambda(policy, lam, episodes, LamEnv, env_cfg, seed_base):
    tot = 0.0
    for e in range(episodes):
        env = LamEnv({**env_cfg, "demand_type": "poisson", "poisson_lam": float(lam)})
        c, _ = rollout(env, policy, seed_base + e)
        tot += c
    return tot / episodes


def cost_across_lambdas(policy, lambdas, episodes=200, env_cfg=None, env_class=None,
                        seed_base=EVAL_SEED_BASE):
    """Mean cost of a (possibly adaptive) policy at each lambda. Returns {lam: mean_cost}."""
    env_cfg = dict(ENV_BASE if env_cfg is None else env_cfg)
    LamEnv = _make_lambda_env_class(env_class or get_env_class())
    return {float(lam): _mean_cost_at_lambda(policy, lam, episodes, LamEnv, env_cfg, seed_base)
            for lam in lambdas}


def best_single_fixed_S(lambdas, episodes=80, env_cfg=None, env_class=None,
                        seed_base=SELECT_SEED_BASE, lo=20, hi=160, step=4,
                        per_echelon=False, rounds=3, verbose=True):
    """The DEPLOYABLE BAR's LEVELS: ONE base-stock across the whole lambda set, chosen with NO
    regime knowledge (min mean cost). Selected on `seed_base` (= SELECT). Uniform by default;
    per_echelon=True coordinate-descends a 4-vector. Returns (S_vec, {lam: select_cost}, mean)."""
    env_cfg = dict(ENV_BASE if env_cfg is None else env_cfg)
    LamEnv = _make_lambda_env_class(env_class or get_env_class())

    def mean_cost_of(S_vec):
        per = {lam: _mean_cost_at_lambda(BaseStockPolicy(S_vec), lam, episodes, LamEnv, env_cfg, seed_base)
               for lam in lambdas}
        return float(np.mean(list(per.values()))), per

    best_S, best_mean, best_per = None, np.inf, None
    for s in range(lo, hi + 1, step):
        m, per = mean_cost_of([float(s)] * 4)
        if m < best_mean:
            best_S, best_mean, best_per = [float(s)] * 4, m, per
    if per_echelon:
        S = list(best_S)
        for _ in range(rounds):
            improved = False
            for i in range(4):
                for s in range(lo, hi + 1, step):
                    cand = list(S); cand[i] = float(s)
                    m, per = mean_cost_of(cand)
                    if m < best_mean - 1e-6:
                        S, best_S, best_mean, best_per, improved = cand, cand, m, per, True
            if not improved:
                break
    if verbose:
        kind = "per-echelon" if per_echelon else "uniform"
        print(f"  BAR levels ({kind}): [{','.join(f'{x:.0f}' for x in best_S)}]  select_mean={best_mean:.1f}")
    return best_S, best_per, best_mean


def per_lambda_oracle_levels(lambdas, episodes=80, env_cfg=None, env_class=None,
                             seed_base=SELECT_SEED_BASE, lo=20, hi=160, step=4,
                             per_echelon=True, rounds=3, verbose=True):
    """The privileged ORACLE's per-lambda LEVELS (knows the regime), per_echelon to match
    DRACO's per-echelon head so the ceiling is a TRUE upper bound. Selected on `seed_base`
    (= SELECT). Returns {lam: S_vec}."""
    out = {}
    for lam in lambdas:
        S_vec, _per, _c = best_single_fixed_S([lam], episodes, env_cfg, env_class, seed_base,
                                              lo=lo, hi=hi, step=step, per_echelon=per_echelon,
                                              rounds=rounds, verbose=False)
        out[float(lam)] = S_vec
        if verbose:
            print(f"    oracle lam={lam:>4g}: S~[{','.join(f'{x:.0f}' for x in S_vec)}]")
    return out


def regime_benchmark(lambdas=None, select_episodes=80, eval_episodes=200, env_cfg=None,
                     env_class=None, tau=None, prior_mean=14.0, bar_per_echelon=False,
                     out_path="results/baselines_regime_v2.json", sterman=None, verbose=True):
    """The FOUR-rung, leakage-free C1 ladder. The TEXTBOOK ordering by cost is
        Oracle <= Bayes <~ Adaptive <= static BAR
    and holds under the CANONICAL retailer-only cost (penalty_at_retailer_only=True). Under the
    DEFAULT penalty-at-EVERY-stage cost, the adaptive rungs (Adaptive, Bayes) BULLWHIP and can
    EXCEED the static BAR -- this is expected (see S2), not a bug. So the HEADLINE comparators are
    the static BAR (bullwhip-free, the hard target) + the per-lambda Oracle: DRACO must BEAT the
    static BAR and APPROACH the Oracle; DRACO-vs-Bayes is a SECONDARY 'beats naive adaptation'
    check. Levels (BAR, Oracle) are SELECTED on SELECT_SEED_BASE; ALL rungs are REPORTED on
    EVAL_SEED_BASE (== the DRACO held-out seeds). Writes per-lambda curves to JSON for c1_stats.py
    and the trainer/eval ref loader."""
    lambdas = [float(l) for l in (HELDOUT_LAMBDAS if lambdas is None else lambdas)]
    env_cfg = dict(ENV_BASE if env_cfg is None else env_cfg)
    h, b = env_cfg["holding_cost"], env_cfg["backorder_cost"]
    LamEnv = _make_lambda_env_class(env_class or get_env_class())

    if verbose:
        print("=" * 76)
        print(f"REGIME-UNCERTAINTY BENCHMARK (4-rung) | lambda in {[f'{l:g}' for l in lambdas]}")
        print(f"  select seeds [{SELECT_SEED_BASE}..+{select_episodes}]  |  "
              f"eval seeds [{EVAL_SEED_BASE}..+{eval_episodes}]  (disjoint -> no in-sample bias)")
        print("=" * 76)
        print("Selecting BAR + per-lambda Oracle levels on SELECT seeds ...")
    bar_levels, _, _ = best_single_fixed_S(lambdas, select_episodes, env_cfg, env_class,
                                           SELECT_SEED_BASE, per_echelon=bar_per_echelon,
                                           verbose=verbose)
    oracle_levels = per_lambda_oracle_levels(lambdas, select_episodes, env_cfg, env_class,
                                             SELECT_SEED_BASE, per_echelon=True, verbose=verbose)

    if verbose:
        print("Scoring all rungs on EVAL seeds ...")
    bayes = make_bayes_rung(tau=tau, h=h, b=b, prior_mean=prior_mean)   # retailer-Bayes + adaptive upstream
    adaptive = AdaptiveForecastPolicy(tau=tau, h=h, b=b)
    rungs = {
        "Oracle": {l: _mean_cost_at_lambda(BaseStockPolicy(oracle_levels[l]), l, eval_episodes,
                                           LamEnv, env_cfg, EVAL_SEED_BASE) for l in lambdas},
        "BAR_static": cost_across_lambdas(BaseStockPolicy(bar_levels), lambdas, eval_episodes,
                                          env_cfg, env_class, EVAL_SEED_BASE),
        "Bayes": cost_across_lambdas(bayes, lambdas, eval_episodes, env_cfg, env_class, EVAL_SEED_BASE),
        "Adaptive": cost_across_lambdas(adaptive, lambdas, eval_episodes, env_cfg, env_class, EVAL_SEED_BASE),
    }
    means = {name: float(np.mean([rungs[name][l] for l in lambdas])) for name in rungs}

    if verbose:
        print("\n  rung          " + "".join(f"{f'lam={l:g}':>12}" for l in lambdas) + f"{'mean':>12}")
        for name in ("BAR_static", "Adaptive", "Bayes", "Oracle"):
            row = "".join(f"{rungs[name][l]:>12.1f}" for l in lambdas)
            print(f"  {name:<14}{row}{means[name]:>12.1f}")
        print("  (Oracle = per-lambda best static base-stock = a validated ceiling; a PROVEN optimum "
              "only under the canonical cost. Adaptive/Bayes are naive-adaptation floors -- bullwhip, S2.)")
        print(f"\n  Headroom DRACO must capture (BAR-Oracle) = {means['BAR_static'] - means['Oracle']:.1f}")
        print(f"  Bayes - BAR = {means['Bayes'] - means['BAR_static']:+.1f}  "
              f"(>0 = naive adaptation BULLWHIPS above the static bar; expected in the per-stage cost)")
        print(f"  DRACO targets: BEAT static BAR={means['BAR_static']:.0f} (the real bar) and "
              f"APPROACH Oracle={means['Oracle']:.0f};  Bayes={means['Bayes']:.0f} is the naive-adaptation FLOOR.")
        if not (means["Oracle"] <= means["BAR_static"] + 1e-6):
            print("  [WARN] Oracle > BAR: the per-lambda oracle should be the cheapest rung; "
                  "re-check the oracle optimization (selection seeds / grid).")
        if sterman is not None:
            per = cost_across_lambdas(sterman, lambdas, eval_episodes, env_cfg, env_class, EVAL_SEED_BASE)
            print(f"  Sterman behavioral floor (mean) = {np.mean(list(per.values())):.1f}")

    try:
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        with open(out_path, "w") as f:
            json.dump({"rungs": rungs, "lambdas": lambdas,
                       # back-compat scalars (old consumers read BAR/CEILING):
                       "BAR": means["BAR_static"], "CEILING": means["Oracle"],
                       "BAYES": means["Bayes"], "ADAPTIVE": means["Adaptive"],
                       "meta": {"select_seed_base": SELECT_SEED_BASE, "eval_seed_base": EVAL_SEED_BASE,
                                "select_episodes": select_episodes, "eval_episodes": eval_episodes,
                                "tau": _per_echelon(DEFAULT_TAU if tau is None else tau, "tau"),
                                "prior_mean": prior_mean, "h": h, "b": b,
                                "bar_levels": bar_levels,
                                "oracle_levels": {str(l): oracle_levels[l] for l in lambdas}}},
                      f, indent=2)
        if verbose:
            print(f"\n  -> wrote {out_path} (feed to c1_stats.summarize and the trainer/eval ref loader)")
    except OSError as e:
        print(f"  (could not write {out_path}: {e})")

    return dict(rungs=rungs, means=means, bar_levels=bar_levels, oracle_levels=oracle_levels)


# ==============================================================================
# CLI
# ==============================================================================
def main():
    ap = argparse.ArgumentParser(description="DRACO beer-game baselines (base-stock + Bayes/adaptive + Sterman).")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("selftest")

    pr = sub.add_parser("regime", help="four-rung C1 ladder (BAR/Adaptive/Bayes/Oracle, leak-free)")
    pr.add_argument("--lambdas", nargs="+", type=float, default=HELDOUT_LAMBDAS)
    pr.add_argument("--select-episodes", type=int, default=80)
    pr.add_argument("--eval-episodes", type=int, default=200)
    pr.add_argument("--prior-mean", type=float, default=14.0)
    pr.add_argument("--bar-per-echelon", action="store_true",
                    help="coordinate-descend the static BAR per-echelon (tighter, still static)")
    pr.add_argument("--sterman", nargs=4, type=float, default=None,
                    metavar=("THETA", "ALPHA_S", "BETA", "S_PRIME"),
                    help="also report the Sterman behavioral floor")
    pr.add_argument("--penalty-at-retailer-only", action="store_true",
                    help="canonical Clark-Scarf cost (the provable-optimum C1 variant); writes "
                         "results/baselines_regime_v2_canonical.json")

    pc = sub.add_parser("canonical", help="provable optimum on the CANONICAL serial model (M3 anchor)")
    pc.add_argument("--lambdas", nargs="+", type=float, default=HELDOUT_LAMBDAS)
    pc.add_argument("--L", type=int, default=4)
    pc.add_argument("--h", type=float, default=0.5)
    pc.add_argument("--b", type=float, default=1.0)

    pv = sub.add_parser("validate-canonical",
                        help="validate the ENV canonical oracle is the PROVABLE optimum (M3.4/M3.5)")
    pv.add_argument("--lambdas", nargs="+", type=float, default=HELDOUT_LAMBDAS)
    pv.add_argument("--L", type=int, default=4, help="idealized lead = env order_lead(2) + ship_lead(2)")
    pv.add_argument("--select-episodes", type=int, default=60)
    pv.add_argument("--eval-episodes", type=int, default=120)
    pv.add_argument("--step", type=int, default=8, help="grid step for the unimodality sweep")
    pv.add_argument("--no-per-echelon", action="store_true", help="uniform env oracle (faster)")

    ps = sub.add_parser("serial", help="idealized serial base-stock levels (theory anchor)")
    ps.add_argument("--lam", type=float, default=8.0)
    ps.add_argument("--L", type=int, default=4, help="use 4 to mimic env order(2)+ship(2)")
    ps.add_argument("--h", type=float, default=0.5)
    ps.add_argument("--b", type=float, default=1.0)

    po = sub.add_parser("optimize", help="coordinate-descent base-stock ON the env")
    po.add_argument("--regimes", nargs="+", default=["poisson", "black_swan", "extreme_chaos"])
    po.add_argument("--episodes", type=int, default=40)

    pe = sub.add_parser("eval", help="evaluate optimized base-stock + Sterman ON the env")
    pe.add_argument("--regimes", nargs="+", default=["poisson", "black_swan", "extreme_chaos"])
    pe.add_argument("--seeds", type=int, default=20)
    pe.add_argument("--cs-levels", nargs=4, type=float, default=None,
                    metavar=("S_RET", "S_WHO", "S_DIS", "S_MFR"))
    pe.add_argument("--cs-opt-episodes", type=int, default=30)
    pe.add_argument("--sterman", nargs=4, type=float, default=[0.25, 0.36, 0.34, 17.0],
                    metavar=("THETA", "ALPHA_S", "BETA", "S_PRIME"))

    args = ap.parse_args()

    if args.cmd == "selftest":
        selftest()
    elif args.cmd == "regime":
        sterman = StermanPolicy(*args.sterman) if args.sterman else None
        env_cfg, out = dict(ENV_BASE), "results/baselines_regime_v2.json"
        if args.penalty_at_retailer_only:
            env_cfg["penalty_at_retailer_only"] = True              # canonical Clark-Scarf cost
            out = "results/baselines_regime_v2_canonical.json"
        regime_benchmark(args.lambdas, args.select_episodes, args.eval_episodes, env_cfg=env_cfg,
                         prior_mean=args.prior_mean, bar_per_echelon=args.bar_per_echelon,
                         sterman=sterman, out_path=out)
    elif args.cmd == "canonical":
        for l in args.lambdas:
            optimize_canonical_serial(l, args.L, args.h, args.b)
    elif args.cmd == "validate-canonical":
        validate_canonical(args.lambdas, L=args.L, select_episodes=args.select_episodes,
                           eval_episodes=args.eval_episodes, step=args.step,
                           per_echelon=not args.no_per_echelon)
    elif args.cmd == "serial":
        S, c = coord_descent_serial(args.lam, args.L, args.h, args.b)
        print(f"Optimized serial base-stock (idealized, lam={args.lam}, L={args.L}, "
              f"h={args.h}, b={args.b}):")
        print(f"  S* = [{', '.join(f'{x:.0f}' for x in S)}]   mean episode cost = {c:.1f}")
    elif args.cmd == "optimize":
        print("=" * 70)
        print("Per-echelon base-stock optimized ON THE ENV")
        print("=" * 70)
        for reg in args.regimes:
            optimize_basestock_on_env(reg, episodes=args.episodes)
    elif args.cmd == "eval":
        th, al, be, sp = args.sterman
        sterman = StermanPolicy(theta=th, alpha_S=al, beta=be, S_prime=sp)
        if args.cs_levels is not None:
            evaluate({"BaseStock": OptimizedBaseStockPolicy(list(args.cs_levels)), "Sterman": sterman},
                     args.regimes, n_seeds=args.seeds)
        else:
            print("(no --cs-levels given: optimizing base-stock per regime on the env first)\n")
            for reg in args.regimes:
                S, _ = optimize_basestock_on_env(reg, episodes=args.cs_opt_episodes)
                evaluate({"BaseStock": OptimizedBaseStockPolicy(S), "Sterman": sterman},
                         [reg], n_seeds=args.seeds)


if __name__ == "__main__":
    main()