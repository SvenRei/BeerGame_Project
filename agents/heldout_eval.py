"""
heldout_eval.py -- held-out-lambda evaluation for DRACO (the C1 gate).

Scores the trained policy on a set of STATIONARY poisson regimes whose rate the agent
does NOT know in advance, against the two reference numbers from baselines.py:

    BAR      = best single fixed base-stock committed across the lambda set (no regime
               knowledge)        -> the deployable policy DRACO must BEAT.
    CEILING  = per-lambda oracle (privileged: knows the regime)
                                 -> what DRACO APPROACHES but cannot beat.

The headline metric Eval_lambda/Gap_Recovered = (BAR - DRACO) / (BAR - CEILING):
    >= 0  : DRACO beats the deployable fixed policy (the C1 claim),
       1  : DRACO matches the privileged oracle,
    <  0  : DRACO is worse than the fixed bar (no regime inference happening).

This module does NOT reimplement the rollout. It calls the trainer's own `run_episode`
closure (deterministic=True), so the policy evaluated here is byte-for-byte the trained
policy. The only thing it owns is which envs to run and how to aggregate.

INTEGRATION (train_draco_v4.py) -- see the three-line insert at the bottom of this file.

CRN: episodes are reset at seed_base + e (== baselines.py SEED_BASE + e), so per-lambda
costs are directly comparable to `python baselines.py regime`. Keep HELDOUT_LAMBDAS and
SEED_BASE identical to baselines.py.
"""
import numpy as np

# MUST match baselines.py (the bar/ceiling are computed there on the same lambdas/seeds).
HELDOUT_LAMBDAS = [6.0, 10.0, 14.0, 18.0, 22.0]
SEED_BASE = 100000


def make_heldout_envs(env_cls, env_cfg, lambdas=HELDOUT_LAMBDAS):
    """Build one STATIONARY-lambda poisson env per held-out regime, ONCE (reused every
    eval round via reset inside run_episode). `env_cls` is DemandRandomizedBeerGame:
    lo=hi=lam, p_shift=0 -> a fixed-rate poisson episode, the same demand baselines.py
    scores the fixed/oracle policies on at the same seeds."""
    return {float(lam): env_cls({**env_cfg, "demand_type": "poisson"},
                                lam_lo=float(lam), lam_hi=float(lam), p_shift=0.0)
            for lam in lambdas}


def run_heldout_eval(run_episode, heldout_envs, base_seed, episodes=20,
                     fixed_ref=4726.0, oracle_ref=2202.0, seed_base=SEED_BASE):
    """Run `episodes` deterministic episodes per held-out lambda via the trainer's
    run_episode, and return a dict of W&B metrics.

    run_episode : the trainer closure, signature
                  run_episode(ep, target_env, collect, deterministic) ->
                      (buf, ep_cost, ep_costs, msgs_log, (s_mean, order_mean))
    fixed_ref / oracle_ref : the BAR / CEILING from `python baselines.py regime`
                  (defaults are the uniform-S numbers for HELDOUT_LAMBDAS on ENV_BASE;
                  regenerate and pass yours if h/b/horizon or the lambda set differ)."""
    per_lambda, log = {}, {}
    for lam, env in heldout_envs.items():
        costs, s_means = [], []
        for e in range(episodes):
            # run_episode resets at base_seed + ep; choose ep so the reset seed is
            # seed_base + e (== baselines.py), giving CRN-comparable per-lambda costs.
            ep = (seed_base - base_seed) + e
            _, ep_cost, _, _, (s_mean, _order_mean) = run_episode(
                ep, env, collect=False, deterministic=True)
            costs.append(ep_cost)
            s_means.append(s_mean)
        per_lambda[lam] = float(np.mean(costs))
        log[f"Eval_lambda/{lam:g}_Cost"] = float(np.mean(costs))
        log[f"Eval_lambda/{lam:g}_S_mean"] = float(np.mean(s_means))

    mean_cost = float(np.mean(list(per_lambda.values())))
    denom = max(1e-6, float(fixed_ref) - float(oracle_ref))
    log["Eval_lambda/Mean_Cost"] = mean_cost                       # the number DRACO drives down
    log["Eval_lambda/Gap_Recovered"] = (float(fixed_ref) - mean_cost) / denom
    log["Eval_lambda/vs_Fixed_Pct"] = 100.0 * (float(fixed_ref) - mean_cost) / max(1e-6, float(fixed_ref))
    return log


# ==============================================================================
# INTEGRATION SNIPPET for train_draco_v4.py  (copy the marked lines in)
# ------------------------------------------------------------------------------
# 1) Imports -- next to the other `from agents.rl.draco_v4 import (...)` line:
#
#       from heldout_eval import make_heldout_envs, run_heldout_eval
#    (or `from agents.rl.heldout_eval import ...` if you drop this file in agents/rl/)
#
# 2) Build the envs ONCE -- right after eval_env_poisson / eval_env_ood are built
#    (~line 72), so they live for the whole run:
#
#       heldout_envs   = make_heldout_envs(DemandRandomizedBeerGame, env_cfg)
#       heldout_every  = cfg.agent.get("heldout_every", 200)
#       heldout_eps    = cfg.agent.get("heldout_episodes", 20)
#       heldout_fixed  = cfg.agent.get("heldout_fixed_ref", 4268.0)   # from `baselines.py regime`
#       heldout_oracle = cfg.agent.get("heldout_oracle_ref", 2182.0)  # from `baselines.py regime`
#
# 3) Score it -- inside the eval block, BEFORE `wandb.log(log)` (~line 274). NOTE: this
#    must sit INSIDE the `for ep` loop. In the file you uploaded, the whole block from
#    `cost_hist.append(...)` (line 238) down is indented at 4 spaces == OUTSIDE the loop,
#    so eval (and this) would run once at the end -- re-indent that block into the loop:
#
#       if ep > warm_up and ep % heldout_every == 0:
#           log.update(run_heldout_eval(
#               run_episode, heldout_envs, base_seed, heldout_eps,
#               fixed_ref=heldout_fixed, oracle_ref=heldout_oracle))
#
# Read Eval_lambda/Gap_Recovered on the W&B panel: crossing 0 == the C1 result exists.
# ==============================================================================


if __name__ == "__main__":
    # Self-test the aggregation/metric math with a mock run_episode (no torch/env needed).
    def _mock_run_episode(cost):
        # returns a callable matching the trainer signature, ignoring inputs
        def f(ep, env, collect, deterministic):
            return None, cost + 0.0 * ep, None, None, (cost / 50.0, cost / 60.0)
        return f

    envs = {6.0: "e6", 10.0: "e10", 14.0: "e14", 18.0: "e18", 22.0: "e22"}

    # Case A: DRACO exactly at the oracle ceiling -> Gap_Recovered == 1.0
    log = run_heldout_eval(_mock_run_episode(2202.0), envs, base_seed=1000, episodes=3,
                           fixed_ref=4726.0, oracle_ref=2202.0)
    assert abs(log["Eval_lambda/Gap_Recovered"] - 1.0) < 1e-6, log["Eval_lambda/Gap_Recovered"]

    # Case B: DRACO exactly at the fixed bar -> Gap_Recovered == 0.0
    log = run_heldout_eval(_mock_run_episode(4726.0), envs, base_seed=1000, episodes=3,
                           fixed_ref=4726.0, oracle_ref=2202.0)
    assert abs(log["Eval_lambda/Gap_Recovered"] - 0.0) < 1e-6, log["Eval_lambda/Gap_Recovered"]

    # Case C: DRACO between the two -> 0 < Gap_Recovered < 1, vs_Fixed_Pct > 0
    log = run_heldout_eval(_mock_run_episode(3000.0), envs, base_seed=1000, episodes=3,
                           fixed_ref=4726.0, oracle_ref=2202.0)
    assert 0.0 < log["Eval_lambda/Gap_Recovered"] < 1.0
    assert log["Eval_lambda/vs_Fixed_Pct"] > 0.0
    assert log["Eval_lambda/Mean_Cost"] == 3000.0
    assert log["Eval_lambda/6_Cost"] == 3000.0 and "Eval_lambda/22_S_mean" in log

    # Case D: DRACO worse than the bar -> Gap_Recovered < 0 (no regime inference)
    log = run_heldout_eval(_mock_run_episode(5500.0), envs, base_seed=1000, episodes=3,
                           fixed_ref=4726.0, oracle_ref=2202.0)
    assert log["Eval_lambda/Gap_Recovered"] < 0.0

    print("heldout_eval self-test PASS (Gap_Recovered: oracle=1.0, bar=0.0, mid in (0,1), worse<0)")