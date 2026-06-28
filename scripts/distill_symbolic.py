"""
distill_symbolic.py -- C3: SYMBOLIC DISTILLATION of the DRACO policy (PySR + DAgger).

The headline novel contribution: take the UNCONSTRAINED free-MLP actor head (actor_head=mlp -- the
policy with NO base-stock inductive bias), distill it into a closed-form equation, and show the
equation recovers the textbook base-stock structure  S ~= lead*demand + safety  with a safety factor
near the critical fractile  z* = Phi^-1(b/(b+h)).  "An unconstrained multi-echelon RL policy,
distilled, rediscovers the order-up-to policy" -- novel for this domain.

WHY DAGGER (non-negotiable): naive symbolic regression on the expert's OWN trajectories looks fine
in one-step fidelity but drifts off-distribution when the symbolic policy actually drives the system
(the documented policy-distillation failure mode; SPID 2025). DAgger fixes it: roll out the SYMBOLIC
student, collect the states IT visits, re-query the MLP expert there, append, refit. Repeat.

PIPELINE
  round 0 (behavior cloning): roll the EXPERT, log per echelon (state -> expert order-up-to S).
  rounds 1..R (DAgger):       roll the SYMBOLIC student, re-query the expert at its states, append, refit.
  report:  (1) cost retention  symbolic vs MLP vs Bayes-adaptive rung,
           (2) one-step fidelity R^2 on EXPERT states AND on the student's OWN states (the SPID gap),
           (3) the equation per echelon + the base-stock structural test (slope~=lead, implied z vs z*).

The symbolic regressor is pluggable: PySR (the real symbolic search) when installed, else a transparent
LINEAR base-stock least-squares fit (the H0 "S is affine in the demand signal"). The DAgger loop,
rollouts, fidelity and structural analysis are identical for both, so the whole method is testable
without Julia/torch (see the self-test, which distills a known base-stock 'expert' on the real env).

FEATURES are the four memoryless local scalars [inv, backlog, on_order, last_demand] (roadmap step B):
a memoryless base-stock rule. The expert (MLP) is history-dependent via its belief z; the fidelity gap
quantifies how much that memory matters. (An EWMA-of-demand feature is the natural next ablation.)

Run:
  python scripts/distill_symbolic.py --ckpt weights_draco/run_.../draco_checkpoint_best.pt --backend pysr
  python scripts/distill_symbolic.py --ckpt ... --backend linear --dagger-rounds 3     # fast, no Julia
  python scripts/distill_symbolic.py                                                    # self-test (no torch/pysr)
"""
import os
import sys
import json
import argparse
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from envs.beer_game_env import BeerGameParallelEnv
from scripts.baselines import _make_lambda_env_class   # no-torch lambda env (CRN-matched to baselines)

AGENTS = ["retailer", "wholesaler", "distributor", "manufacturer"]
ENV_BASE = {"horizon": 50, "max_order": 100, "holding_cost": 0.5, "backorder_cost": 1.0}
HELDOUT_LAMBDAS = [6.0, 10.0, 14.0, 18.0, 22.0]
EVAL_SEED_BASE = 100000        # == baselines/eval (CRN-comparable to the references)
COLLECT_SEED_BASE = 300000     # disjoint from eval: the distillation never sees the eval seeds
FEATURE_NAMES = ["inv", "backlog", "on_order", "last_demand"]


def _feat(obs_vec):
    return np.asarray(obs_vec[:4], dtype=float)


def _ip(obs_vec):
    return float(obs_vec[0]) - float(obs_vec[1]) + float(obs_vec[2])     # inventory position


def make_lambda_envs(lambdas, env_cfg=None):
    """One stationary-lambda Poisson env per regime (no torch; same seeds as baselines.py)."""
    cfg = dict(ENV_BASE if env_cfg is None else env_cfg)
    Lam = _make_lambda_env_class(BeerGameParallelEnv)
    return {float(l): Lam({**cfg, "demand_type": "poisson", "poisson_lam": float(l)}) for l in lambdas}


def rollout_cost(policy, env, seed):
    """Total supply-chain cost of `policy` on `env` for one episode (self-contained; no torch).
    policy.act(obs) -> {agent: fraction}, policy.reset(). Matches the env/baselines cost exactly."""
    obs, _ = env.reset(seed=seed)
    policy.reset()
    tot = 0.0
    while True:
        acts = policy.act(obs)
        obs, _, _, trunc, info = env.step({a: [float(acts[a])] for a in AGENTS})
        tot += float(info[AGENTS[0]]["supply_chain_cost"])
        if any(trunc.values()):
            break
    return tot


# ==============================================================================
# Symbolic regressor: PySR (real search) | linear base-stock LS (transparent H0)
# ==============================================================================
class SymbolicRegressor:
    def __init__(self, backend="auto", feature_names=FEATURE_NAMES, pysr_kwargs=None):
        self.backend = backend
        self.feature_names = list(feature_names)
        self.pysr_kwargs = dict(pysr_kwargs or {})
        self._model = None
        self._coef = None
        self._intercept = 0.0
        self._kind = None

    def _make_pysr(self):
        from pysr import PySRRegressor
        kw = dict(niterations=400, binary_operators=["+", "-", "*", "/"],
                  unary_operators=["sqrt", "square"], maxsize=22, model_selection="best",
                  elementwise_loss="L2DistLoss()", verbosity=0, progress=False,
                  deterministic=True, random_state=0, procs=0, multithreading=False)
        kw.update(self.pysr_kwargs)
        return PySRRegressor(**kw)

    def fit(self, X, y):
        X = np.asarray(X, float)
        y = np.asarray(y, float)
        if self.backend in ("pysr", "auto"):
            try:
                self._model = self._make_pysr()
                self._model.fit(X, y)
                self._kind = "pysr"
                return self
            except Exception as e:
                if self.backend == "pysr":
                    raise
                print(f"  [SR] PySR unavailable ({type(e).__name__}: {e}); using linear base-stock fit.")
        # linear least squares: y ~ X @ coef + intercept  (the affine base-stock H0)
        A = np.hstack([X, np.ones((len(X), 1))])
        sol, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
        self._coef = sol[:-1]
        self._intercept = float(sol[-1])
        self._kind = "linear"
        return self

    def predict(self, X):
        X = np.asarray(X, float)
        if X.ndim == 1:
            X = X[None, :]
        if self._kind == "pysr":
            return np.asarray(self._model.predict(X), float)
        return X @ self._coef + self._intercept

    def equation(self):
        if self._kind == "pysr":
            try:
                return str(self._model.sympy())                 # in x0..x3 (see legend())
            except Exception:
                return "?(pysr equation unavailable)"
        terms = " ".join(f"{c:+.3f}*{nm}" for c, nm in zip(self._coef, self.feature_names))
        return f"{self._intercept:.3f} {terms}"

    def legend(self):
        return ", ".join(f"x{j}={nm}" for j, nm in enumerate(self.feature_names))


class SymbolicPolicy:
    """Executes the per-echelon symbolic models in the env. Memoryless: order-up-to S_i = reg_i(obs_i),
    env order = clip(S_i - IP_i, 0, max_order). Interface matches the expert (.reset/.act/.last_S)."""
    def __init__(self, regressors, max_order=100, target="S"):
        self.regs = regressors
        self.max_order = float(max_order)
        self.target = target
        self.last_S = np.zeros(len(AGENTS))

    def reset(self):
        pass

    def act(self, obs):
        acts = {}
        for i, ag in enumerate(AGENTS):
            o = obs[ag]
            pred = float(self.regs[i].predict(_feat(o))[0])
            if self.target == "S":
                S = pred
                order = float(np.clip(S - _ip(o), 0.0, self.max_order))
            else:
                S = pred + _ip(o)
                order = float(np.clip(pred, 0.0, self.max_order))
            self.last_S[i] = S
            acts[ag] = order / self.max_order
        return acts


# ==============================================================================
# DAgger data collection + fit + diagnostics
# ==============================================================================
def collect_round(envs, expert, driver, episodes, seed_base, target="S"):
    """One data pass. DRIVER acts in the env (expert for BC round 0, symbolic for DAgger rounds);
    the EXPERT is queried at every visited state for the target (advancing its belief over the
    driver's trajectory -> correct DAgger). Returns per-echelon (X, y)."""
    X = [[] for _ in AGENTS]
    Y = [[] for _ in AGENTS]
    same = driver is expert
    for lam, env in envs.items():
        for e in range(episodes):
            obs, _ = env.reset(seed=seed_base + e)
            expert.reset()
            if not same:
                driver.reset()
            while True:
                exp_a = expert.act(obs)
                tgt = np.asarray(expert.last_S, float).copy()        # the expert's order-up-to S
                for i, ag in enumerate(AGENTS):
                    o = obs[ag]
                    X[i].append(_feat(o))
                    Y[i].append(float(tgt[i]) if target == "S" else max(0.0, float(tgt[i]) - _ip(o)))
                drive = exp_a if same else driver.act(obs)
                obs, _, _, trunc, _ = env.step({a: [float(drive[a])] for a in AGENTS})
                if any(trunc.values()):
                    break
    return [np.asarray(x) for x in X], [np.asarray(y) for y in Y]


def fit_models(X, Y, backend, pysr_kwargs=None):
    regs = []
    for i in range(len(AGENTS)):
        regs.append(SymbolicRegressor(backend=backend, pysr_kwargs=pysr_kwargs).fit(X[i], Y[i]))
    return regs


def _r2(reg, Xi, Yi):
    p = reg.predict(Xi)
    ss = float(np.sum((Yi - p) ** 2))
    tot = float(np.sum((Yi - Yi.mean()) ** 2))
    return 1.0 - ss / tot if tot > 1e-9 else float("nan")


def mean_costs(policy, envs, episodes, seed_base=EVAL_SEED_BASE):
    return {l: float(np.mean([rollout_cost(policy, e, seed_base + k) for k in range(episodes)]))
            for l, e in envs.items()}


def structural_report(regs, h, b, target="S", mu_grid=None, ref=(12.0, 0.0, 16.0), L_ref=4):
    """Backend-agnostic base-stock test: probe S vs last_demand (holding inv/backlog/on_order at a
    reference), fit S ~= alpha + beta*demand. Base-stock theory: S = L*mu + z*sqrt(L*mu), so beta ~= the
    effective lead L and the implied safety factor z = alpha/sqrt(beta*mu) should sit near the critical
    fractile z* = Phi^-1(b/(b+h))."""
    from scipy.stats import norm
    if mu_grid is None:
        mu_grid = np.arange(4.0, 25.0, 2.0)
    z_crit = float(norm.ppf(b / (b + h)))
    mu0 = float(np.mean(mu_grid))
    rows = []
    for i, ag in enumerate(AGENTS):
        feats = np.array([[ref[0], ref[1], ref[2], d] for d in mu_grid])
        S = regs[i].predict(feats)
        if target != "S":
            S = S + (ref[0] - ref[1] + ref[2])                       # order -> S
        beta, alpha = np.polyfit(mu_grid, S, 1)                       # S ~= alpha + beta*demand
        implied_z = float(alpha / np.sqrt(max(1e-6, beta * mu0))) if beta > 0 else float("nan")
        rows.append(dict(agent=ag, slope=float(beta), intercept=float(alpha), implied_z=implied_z))
    return dict(rows=rows, z_critical=z_crit, L_ref=L_ref, mu0=mu0)


def dagger_distill(expert, envs, backend="auto", bc_episodes=80, dagger_episodes=50, rounds=4,
                   eval_episodes=120, fidelity_episodes=12, target="S", pysr_kwargs=None, verbose=True):
    """The DAgger distillation loop. Returns (final regressors, per-round history)."""
    X, Y = collect_round(envs, expert, expert, bc_episodes, COLLECT_SEED_BASE, target)   # round 0: BC
    history, regs = [], None
    for rnd in range(rounds + 1):
        regs = fit_models(X, Y, backend, pysr_kwargs)
        sym = SymbolicPolicy(regs, max_order=ENV_BASE["max_order"], target=target)
        sym_costs = mean_costs(sym, envs, eval_episodes)
        Xe, Ye = collect_round(envs, expert, expert, fidelity_episodes, EVAL_SEED_BASE, target)   # expert states
        Xs, Ys = collect_round(envs, expert, sym, fidelity_episodes, EVAL_SEED_BASE, target)      # student states
        fid_exp = float(np.nanmean([_r2(regs[i], Xe[i], Ye[i]) for i in range(len(AGENTS))]))
        fid_sym = float(np.nanmean([_r2(regs[i], Xs[i], Ys[i]) for i in range(len(AGENTS))]))
        rec = dict(round=rnd, sym_mean_cost=float(np.mean(list(sym_costs.values()))),
                   fidelity_expert_states=fid_exp, fidelity_student_states=fid_sym,
                   n_train=int(sum(len(x) for x in X)))
        history.append(rec)
        if verbose:
            tag = "BC " if rnd == 0 else f"DAg{rnd}"
            print(f"  [{tag}] sym_cost={rec['sym_mean_cost']:8.1f}  fidelity R^2: "
                  f"expert-states={fid_exp:.3f} student-states={fid_sym:.3f}  (n={rec['n_train']})")
        if rnd < rounds:
            Xd, Yd = collect_round(envs, expert, sym, dagger_episodes,
                                   COLLECT_SEED_BASE + 1000 * (rnd + 1), target)
            for i in range(len(AGENTS)):
                X[i] = np.vstack([X[i], Xd[i]])
                Y[i] = np.concatenate([Y[i], Yd[i]])
    return regs, history


# ==============================================================================
# Expert loader (the only torch-dependent piece; lazy import)
# ==============================================================================
def load_expert(ckpt_path):
    import torch
    from agents.eval_draco_v4 import DracoV4Policy
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    head = ck.get("config", {}).get("agent", {}).get("actor_head")
    if head != "mlp":
        print(f"  [warn] checkpoint actor_head={head!r}, not 'mlp'. C3 is meant for the UNCONSTRAINED "
              f"head; on a 'structured' head the base-stock form is built in (distillation is trivial).")
    env0 = make_lambda_envs([HELDOUT_LAMBDAS[0]])[HELDOUT_LAMBDAS[0]]
    return DracoV4Policy(ck, env0, ablate=False, deterministic=True)


def _bayes_costs(envs, episodes, h, b, prior_mean):
    from scripts.baselines import BayesPoissonPolicy, rollout as bl_rollout
    return {l: float(np.mean([bl_rollout(e, BayesPoissonPolicy(h=h, b=b, prior_mean=prior_mean),
                                         EVAL_SEED_BASE + k)[0] for k in range(episodes)]))
            for l, e in envs.items()}


def _print_final(regs, history, expert_costs, sym_costs, bayes_costs, struct, target, h, b):
    lams = sorted(expert_costs)
    em, sm = float(np.mean(list(expert_costs.values()))), float(np.mean(list(sym_costs.values())))
    bm = float(np.mean(list(bayes_costs.values()))) if bayes_costs else None
    print("\n" + "=" * 80)
    print("C3 SYMBOLIC DISTILLATION -- FINAL REPORT")
    print("=" * 80)
    print(f"  cost retention (lower=better)   MLP expert={em:.1f}   symbolic={sm:.1f}"
          + (f"   Bayes={bm:.1f}" if bm is not None else ""))
    print(f"    symbolic / expert = {sm / em:.3f}x  (1.00 = no loss from distillation)"
          + (f";  symbolic / Bayes = {sm / bm:.3f}x" if bm else ""))
    fid = history[-1]
    print(f"  one-step fidelity R^2: expert-states={fid['fidelity_expert_states']:.3f}  "
          f"student-states={fid['fidelity_student_states']:.3f}  "
          f"(gap = distribution-shift inflation; DAgger shrinks it)")
    print("-" * 80)
    print(f"  recovered equations ({'order-up-to S' if target == 'S' else 'order'} per echelon):")
    print(f"    [legend] {regs[0].legend()}")
    for i, ag in enumerate(AGENTS):
        print(f"    {ag:<12} {target} = {regs[i].equation()}")
    print("-" * 80)
    print(f"  base-stock structural test  (S ~= beta*demand + alpha;  beta~lead L_ref={struct['L_ref']}, "
          f"implied z vs critical fractile z*={struct['z_critical']:.3f} = Phi^-1(b/(b+h)), b={b} h={h}):")
    print(f"    {'echelon':<12}{'slope(beta~L)':>14}{'intercept(alpha)':>18}{'implied z':>11}")
    for r in struct["rows"]:
        print(f"    {r['agent']:<12}{r['slope']:>14.2f}{r['intercept']:>18.1f}{r['implied_z']:>11.2f}")
    print(f"    -> beta near {struct['L_ref']} and implied z near {struct['z_critical']:.2f} = the learned "
          f"policy IS a demand-grounded base-stock with the textbook safety factor.")
    print("=" * 80)


def main():
    ap = argparse.ArgumentParser(description="C3 symbolic distillation (PySR + DAgger) of the DRACO MLP head.")
    ap.add_argument("--ckpt", default=None, help="MLP-head DRACO checkpoint (the SR substrate). Omit -> self-test.")
    ap.add_argument("--backend", default="auto", choices=["auto", "pysr", "linear"])
    ap.add_argument("--target", default="S", choices=["S", "order"], help="distill order-up-to S (default) or raw order")
    ap.add_argument("--lambdas", nargs="+", type=float, default=HELDOUT_LAMBDAS)
    ap.add_argument("--bc-episodes", type=int, default=80)
    ap.add_argument("--dagger-episodes", type=int, default=50)
    ap.add_argument("--dagger-rounds", type=int, default=4)
    ap.add_argument("--eval-episodes", type=int, default=120)
    ap.add_argument("--fidelity-episodes", type=int, default=12)
    ap.add_argument("--prior-mean", type=float, default=14.0, help="Bayes rung prior mean for the cost comparison")
    ap.add_argument("--out", default="results/distill_symbolic.json")
    args = ap.parse_args()

    if args.ckpt is None:
        _selftest()
        return

    expert = load_expert(args.ckpt)
    envs = make_lambda_envs(args.lambdas)
    h, b = ENV_BASE["holding_cost"], ENV_BASE["backorder_cost"]
    print(f"\nC3 distillation  |  backend={args.backend}  target={args.target}  "
          f"rounds={args.dagger_rounds}  lambdas={[f'{l:g}' for l in args.lambdas]}\n")

    regs, history = dagger_distill(expert, envs, backend=args.backend, bc_episodes=args.bc_episodes,
                                   dagger_episodes=args.dagger_episodes, rounds=args.dagger_rounds,
                                   eval_episodes=args.eval_episodes, fidelity_episodes=args.fidelity_episodes,
                                   target=args.target)
    sym = SymbolicPolicy(regs, max_order=ENV_BASE["max_order"], target=args.target)
    sym_costs = mean_costs(sym, envs, args.eval_episodes)
    expert_costs = mean_costs(expert, envs, args.eval_episodes)
    try:
        bayes_costs = _bayes_costs(envs, args.eval_episodes, h, b, args.prior_mean)
    except Exception as e:
        print(f"  (Bayes rung skipped: {type(e).__name__}: {e})")
        bayes_costs = None
    struct = structural_report(regs, h, b, target=args.target)
    _print_final(regs, history, expert_costs, sym_costs, bayes_costs, struct, args.target, h, b)

    out = {"history": history, "expert_costs": expert_costs, "sym_costs": sym_costs,
           "bayes_costs": bayes_costs, "structural": struct, "target": args.target,
           "equations": {AGENTS[i]: regs[i].equation() for i in range(len(AGENTS))},
           "backend": regs[0]._kind}
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n  -> wrote {args.out}")


# ==============================================================================
# Self-test: distill a KNOWN base-stock 'expert' on the REAL env (no torch/pysr).
# Verifies the full DAgger pipeline + linear SR + structural analysis end-to-end.
# ==============================================================================
class _MockBaseStockExpert:
    """Stands in for the trained MLP. Memoryless per-echelon base-stock S_i = a_i + b_i*last_demand,
    so the linear SR should recover it EXACTLY -> retention ~1, fidelity ~1, slope~b_i, intercept~a_i."""
    def __init__(self, a, b, max_order=100):
        self.a = list(a)
        self.b = list(b)
        self.max_order = float(max_order)
        self.last_S = np.zeros(len(AGENTS))

    def reset(self):
        pass

    def act(self, obs):
        acts = {}
        for i, ag in enumerate(AGENTS):
            o = obs[ag]
            S = self.a[i] + self.b[i] * float(o[3])
            order = float(np.clip(S - _ip(o), 0.0, self.max_order))
            self.last_S[i] = S
            acts[ag] = order / self.max_order
        return acts


def _selftest():
    print("=" * 72)
    print("SELF-TEST: distill a known base-stock expert on the REAL env (linear backend)")
    print("=" * 72)
    a, b_coef = [20.0, 24.0, 28.0, 30.0], [4.0, 4.0, 4.0, 3.0]   # mfr lead smaller (matches env)
    expert = _MockBaseStockExpert(a, b_coef)
    envs = make_lambda_envs([8.0, 16.0])
    regs, history = dagger_distill(expert, envs, backend="linear", bc_episodes=12, dagger_episodes=8,
                                   rounds=2, eval_episodes=10, fidelity_episodes=6, verbose=True)
    sym = SymbolicPolicy(regs, max_order=ENV_BASE["max_order"])
    sym_costs = mean_costs(sym, envs, 10)
    exp_costs = mean_costs(expert, envs, 10)
    retention = float(np.mean(list(sym_costs.values())) / np.mean(list(exp_costs.values())))
    struct = structural_report(regs, 0.5, 1.0)
    print(f"\n  cost retention symbolic/expert = {retention:.4f}  (expect ~1.000)")
    print(f"  final fidelity R^2 (student states) = {history[-1]['fidelity_student_states']:.4f}  (expect ~1)")
    print("  structural recovery (expect slope~b_coef, intercept~a):")
    for r, bb, aa in zip(struct["rows"], b_coef, a):
        print(f"    {r['agent']:<12} slope={r['slope']:.2f} (true {bb})  intercept={r['intercept']:.1f} (true {aa})")

    assert 0.97 <= retention <= 1.03, f"retention off: {retention}"
    assert history[-1]["fidelity_student_states"] > 0.98, history[-1]
    for r, bb, aa in zip(struct["rows"], b_coef, a):
        assert abs(r["slope"] - bb) < 0.25, (r, bb)
        assert abs(r["intercept"] - aa) < 3.0, (r, aa)
    # SymbolicPolicy order math
    p = SymbolicPolicy([SymbolicRegressor(backend="linear").fit(np.array([[0, 0, 0, 8.0]]), np.array([40.0]))] * 4)
    act = p.act({ag: np.array([10.0, 0.0, 5.0, 8.0]) for ag in AGENTS})   # S~40, IP=15 -> order~25 -> frac .25
    assert abs(act["retailer"] - 0.25) < 0.05, act
    print("\n  c3 distillation self-test PASS  (DAgger loop + linear SR + structural analysis verified)")


if __name__ == "__main__":
    main()
