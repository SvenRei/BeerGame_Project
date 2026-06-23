"""
baselines.py  --  Self-contained base-stock + behavioral baselines for the DRACO
beer-game thesis. No dependency on the analysis scripts.

Two POLICIES you run ON the env (both editable like the agents):

  ClarkScarfPolicy  : per-echelon installation base-stock,
                          order_i = clip(S_i - IP_i, 0, max_order).
                      For a serial system the optimal policy IS an echelon base-stock
                      (Clark & Scarf 1960); per-echelon coordinate descent on the env
                      finds the optimal installation levels (same achievable cost).
                      EDIT: set S directly, or call optimize_basestock_on_env(...).

  StermanPolicy     : Sterman (1989) anchoring-and-adjustment ordering heuristic,
                          O_t = max(0, Lhat + alpha_S * (S_prime - S_eff - beta*SL)).
                      The behavioral bullwhip baseline. EDIT: theta, alpha_S, beta,
                      S_prime (scalar = shared, or length-4 = per echelon).

Plus a THEORY ANCHOR (no env, pure numpy):
  the idealized serial model + Clark-Scarf optimizer (selftest validates it).

IMPORTANT lead-time note: the env's effective per-echelon delay is
  ORDER lead (2) + SHIPPING lead (2) = 4   (manufacturer: order 1 + production 2 = 3).
The idealized serial model only has one shipping lead L, so to make its levels
comparable to the env pass L=4 (e.g. `serial --L 4`). For the number you REPORT,
prefer the levels optimized directly on the env (`optimize`), which need no such
approximation.

Run:
  python scripts/baselines.py selftest
  python scripts/baselines.py serial   --lam 8 --L 4 --h 0.5 --b 1.0
  python scripts/baselines.py optimize --regimes poisson black_swan extreme_chaos
  python scripts/baselines.py eval      --regimes poisson black_swan extreme_chaos --seeds 20
  python scripts/baselines.py eval      --regimes poisson --seeds 20 \
                                        --cs-levels 44 56 64 68 \
                                        --sterman 0.25 0.36 0.34 17
"""
import argparse
import os
import sys
import numpy as np
from collections import deque

AGENTS = ["retailer", "wholesaler", "distributor", "manufacturer"]

# Default env settings (mirror conf/config.yaml `env:`). EDIT here or via the env config.
ENV_BASE = {"horizon": 50, "max_order": 100, "holding_cost": 0.5, "backorder_cost": 1.0}
SEED_BASE = 100000          # held-out eval seed space (disjoint from training seeds)


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
        # supply_chain_cost is the summed system cost this step (same for every agent)
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


class BaseStockPolicy(Policy):
    """Per-echelon installation base-stock: order_i = clip(S_i - IP_i, 0, max_order).
    IP_i = inventory - backlog + on_order  (recoverable from the 4-scalar obs).

    EDIT: S = scalar (same level everywhere) or [S_ret, S_whole, S_dist, S_mfr]."""
    def __init__(self, S):
        self.S = _per_echelon(S, "S")

    def act(self, obs, env):
        return {a: _action(max(0.0, self.S[i] - _ip(obs[a])), env.max_order)
                for i, a in enumerate(AGENTS)}


class ClarkScarfPolicy(BaseStockPolicy):
    """Clark-Scarf-optimal base-stock. Identical *mechanism* to BaseStockPolicy; the
    point is the LEVELS, which are the optimal echelon base-stock for the serial system.

    Get the levels by either:
      * optimize_basestock_on_env(regime, ...)   <- the honest, env-matched number, or
      * coord_descent_serial(lam, L=4, ...)       <- the idealized theory anchor,
    then construct ClarkScarfPolicy(levels). EDIT the levels freely afterwards."""
    pass


class StermanPolicy(Policy):
    """Sterman (1989) anchoring-and-adjustment ordering heuristic -- the behavioral
    bullwhip baseline.

        Lhat_t = theta * Lhat_{t-1} + (1 - theta) * incoming_t      (demand anchor)
        O_t    = max(0, Lhat_t + alpha_S * (S_prime - S_eff - beta * SL))

      incoming_t : last realized incoming order/demand  (obs[3])
      S_eff      : effective inventory = on_hand - backlog  (obs[0] - obs[1])
      SL         : supply line = on-order                  (obs[2])
      S_prime    : desired effective-inventory level (the anchor target)
      alpha_S    : stock-adjustment fraction in [0,1]
      beta       : supply-line weighting in [0,1]
                     beta = 1  -> fully accounts for in-transit orders (stable / rational)
                     beta < 1  -> UNDER-weights the supply line -> over-ordering -> bullwhip
      theta      : demand-forecast smoothing in [0,1]

    Sterman's mean estimates are roughly alpha_S=0.36, beta=0.34, theta=0.25, S_prime=17;
    those defaults reproduce the classic supply-line-underweighting behavior. EDIT any
    parameter as scalar (shared) or length-4 (per echelon). Set beta=1 for the rational
    (non-bullwhip) variant."""
    def __init__(self, theta=0.25, alpha_S=0.36, beta=0.34, S_prime=17.0, L_init=8.0):
        self.theta = _per_echelon(theta, "theta")
        self.alpha_S = _per_echelon(alpha_S, "alpha_S")
        self.beta = _per_echelon(beta, "beta")
        self.S_prime = _per_echelon(S_prime, "S_prime")
        self.L_init = _per_echelon(L_init, "L_init")
        self.Lhat = None

    def reset(self):
        self.Lhat = list(self.L_init)        # forecast anchor, per echelon

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
# Optimize per-echelon base-stock DIRECTLY ON THE ENV (self-contained; replaces the
# old Part A that imported the analysis scripts). This yields the ClarkScarf levels.
# ==============================================================================
def optimize_basestock_on_env(regime, env_cfg=None, env_class=None, episodes=40,
                              seed_base=SEED_BASE, lo=20, hi=120, step=4, rounds=3, S0=None,
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

    S = list(S0) if S0 is not None else [44.0, 44.0, 44.0, 44.0]
    best = cost_of(S)
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
# THEORY ANCHOR: idealized serial model + Clark-Scarf optimizer (pure numpy, no env)
# ------------------------------------------------------------------------------
# Standard serial beer-game, installation base-stock. stages 0..N-1, 0=retailer.
# Single shipping lead L per stage (no separate order/info delay); pass L=4 to mimic
# the env's order(2)+ship(2). Each period: receive due shipments; order = max(0, S-IP);
# ship min(on_hand, incoming+backlog); cost = sum_i (h*on_hand + b*backlog).
# ==============================================================================
def simulate_serial(S, demand, L=2, h=0.5, b=1.0):
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
        cost += float(np.sum(h * on_hand + b * backlog))   # 4. cost
    return cost


def _demand_samples(lam, T, n_eps, seed):
    rng = np.random.default_rng(seed)
    return rng.poisson(lam, size=(n_eps, T))


def serial_cost(S, lam, L, h, b, T=50, n_eps=200, seed=0):
    D = _demand_samples(lam, T, n_eps, seed)
    return float(np.mean([simulate_serial(S, D[e], L, h, b) for e in range(n_eps)]))


def grid_optimize_serial(lam, L, h, b, T=50, n_eps=120, seed=0, lo=0, hi=120, step=4):
    best, bestc, N = None, np.inf, 4
    for s in range(lo, hi + 1, step):
        c = serial_cost([s] * N, lam, L, h, b, T, n_eps, seed)
        if c < bestc:
            best, bestc = [float(s)] * N, c
    return best, bestc


def coord_descent_serial(lam, L, h, b, T=50, n_eps=120, seed=0,
                         lo=0, hi=140, step=2, rounds=4, S0=None):
    N = 4
    S = list(S0) if S0 is not None else list(grid_optimize_serial(lam, L, h, b, T, n_eps, seed)[0])
    bestc = serial_cost(S, lam, L, h, b, T, n_eps, seed)
    for _ in range(rounds):
        improved = False
        for i in range(N):
            for s in range(lo, hi + 1, step):
                cand = list(S); cand[i] = float(s)
                c = serial_cost(cand, lam, L, h, b, T, n_eps, seed)
                if c < bestc - 1e-9:
                    S, bestc, improved = cand, c, True
        if not improved:
            break
    return S, bestc


def selftest():
    """Validate the serial optimizer: CD beats-or-matches grid; optimal levels rise with
    lambda and with b/h; cost is convex in a single stage."""
    print("=" * 78)
    print("SELF-TEST: idealized serial Clark-Scarf base-stock optimizer")
    print("=" * 78)
    L, h, b = 2, 0.5, 1.0
    for lam in (4, 8, 16):
        g, gc = grid_optimize_serial(lam, L, h, b, seed=1)
        s, sc = coord_descent_serial(lam, L, h, b, seed=1, S0=g)
        ok = sc <= gc + 1e-6
        print(f"  lam={lam:2d} | grid S={g[0]:.0f} cost={gc:8.1f} | "
              f"CD S=[{','.join(f'{x:.0f}' for x in s)}] cost={sc:8.1f} | CD<=grid: {ok}")
    s_lo, _ = coord_descent_serial(8, L, 0.5, 0.5, seed=1)   # b/h = 1
    s_hi, _ = coord_descent_serial(8, L, 0.5, 4.0, seed=1)   # b/h = 8
    print(f"  b/h sensitivity @lam=8: mean S(b/h=1)={np.mean(s_lo):.1f} "
          f"< mean S(b/h=8)={np.mean(s_hi):.1f} : {np.mean(s_lo) < np.mean(s_hi)}")
    base = list(coord_descent_serial(8, L, h, b, seed=1)[0])
    xs = list(range(10, 60, 5)); cs = []
    for x in xs:
        c = list(base); c[0] = x
        cs.append(serial_cost(c, 8, L, h, b, seed=1))
    diffs = np.diff(cs)
    convex = np.all(np.diff(diffs) >= -1e-6)
    print(f"  convex in S_0 (2nd diff >= 0): {convex}  | curve min at S_0~{xs[int(np.argmin(cs))]}")
    print("  PASS\n" if convex else "  CHECK convexity\n")


# ==============================================================================
# CLI
# ==============================================================================
def main():
    ap = argparse.ArgumentParser(description="DRACO beer-game baselines (Clark-Scarf + Sterman).")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("selftest")

    ps = sub.add_parser("serial", help="idealized Clark-Scarf levels (theory anchor)")
    ps.add_argument("--lam", type=float, default=8.0)
    ps.add_argument("--L", type=int, default=4, help="use 4 to mimic env order(2)+ship(2)")
    ps.add_argument("--h", type=float, default=0.5)
    ps.add_argument("--b", type=float, default=1.0)

    po = sub.add_parser("optimize", help="coordinate-descent base-stock ON the env -> ClarkScarf levels")
    po.add_argument("--regimes", nargs="+", default=["poisson", "black_swan", "extreme_chaos"])
    po.add_argument("--episodes", type=int, default=40)

    pe = sub.add_parser("eval", help="evaluate ClarkScarf + Sterman ON the env")
    pe.add_argument("--regimes", nargs="+", default=["poisson", "black_swan", "extreme_chaos"])
    pe.add_argument("--seeds", type=int, default=20)
    pe.add_argument("--cs-levels", nargs=4, type=float, default=None,
                    metavar=("S_RET", "S_WHO", "S_DIS", "S_MFR"),
                    help="ClarkScarf per-echelon levels; if omitted, optimized per regime on the env")
    pe.add_argument("--cs-opt-episodes", type=int, default=30,
                    help="episodes/eval when auto-optimizing ClarkScarf levels")
    pe.add_argument("--sterman", nargs=4, type=float, default=[0.25, 0.36, 0.34, 17.0],
                    metavar=("THETA", "ALPHA_S", "BETA", "S_PRIME"),
                    help="Sterman params (shared across echelons)")

    args = ap.parse_args()

    if args.cmd == "selftest":
        selftest()

    elif args.cmd == "serial":
        S, c = coord_descent_serial(args.lam, args.L, args.h, args.b)
        print(f"Clark-Scarf-optimal serial base-stock (idealized, lam={args.lam}, L={args.L}, "
              f"h={args.h}, b={args.b}):")
        print(f"  S* = [{', '.join(f'{x:.0f}' for x in S)}]   mean episode cost = {c:.1f}")

    elif args.cmd == "optimize":
        print("=" * 70)
        print(f"Per-echelon base-stock optimized ON THE ENV (= ClarkScarf levels)")
        print("=" * 70)
        for reg in args.regimes:
            optimize_basestock_on_env(reg, episodes=args.episodes)

    elif args.cmd == "eval":
        th, al, be, sp = args.sterman
        sterman = StermanPolicy(theta=th, alpha_S=al, beta=be, S_prime=sp)
        # Build one ClarkScarf policy per regime (optimal-per-regime is the honest oracle),
        # unless fixed levels were supplied.
        if args.cs_levels is not None:
            cs = ClarkScarfPolicy(list(args.cs_levels))
            evaluate({"ClarkScarf": cs, "Sterman": sterman}, args.regimes, n_seeds=args.seeds)
        else:
            print("(no --cs-levels given: optimizing ClarkScarf per regime on the env first)\n")
            for reg in args.regimes:
                S, _ = optimize_basestock_on_env(reg, episodes=args.cs_opt_episodes)
                evaluate({"ClarkScarf": ClarkScarfPolicy(S), "Sterman": sterman},
                         [reg], n_seeds=args.seeds)


if __name__ == "__main__":
    main()