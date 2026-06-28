"""
run_confirmatory_report.py -- the LOCKED, preregistered analyzer (review #2/#9/#10). Torch-free.

It executes PREREGISTRATION.md end-to-end so the headline numbers come from one immutable script,
not ad-hoc notebooks:
  * loads the per-seed DRACO dumps produced by `eval_draco_v4.py --dump-c1`
        seed{S}.json     -> {lambda: mean_cost}
        seed{S}_bw.json  -> {lambda: {echelon: BW_cum}}
  * checks the run matches the prereg (>=10 seeds, lambda set == the refs);
  * C1 headline via c1_stats.summarize (Gap [95% CI], IQM, perf profile, Gap<=1, vs-Bayes floor);
  * a BULLWHIP TABLE (DRACO from the dumps; static-BAR + Bayes computed offline on the env);
  * optional COMMUNICATION analysis: CRN-paired Wilcoxon + TOST(+/-band) + Holm across topologies;
  * writes results/confirmatory_report.json.

Run:
  python scripts/run_confirmatory_report.py --draco-dir results/draco_c1 --refs results/baselines_regime_v2.json
  python scripts/run_confirmatory_report.py ... --comm results/c2_nocomm results/c2_full results/c2_neighbor
  python scripts/run_confirmatory_report.py            # self-test on synthetic dumps (no torch; uses the env)
"""
import os
import sys
import json
import glob
import argparse
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.c1_stats import (load_rungs, mean_refs, load_draco_dir, summarize, print_report,
                              paired, tost, compare_many, bootstrap_ci)
from scripts.baselines import BaseStockPolicy, make_bayes_rung, _make_lambda_env_class
from envs.beer_game_env import BeerGameParallelEnv

AGENTS = ["retailer", "wholesaler", "distributor", "manufacturer"]
ENV_BASE = {"horizon": 50, "max_order": 100, "holding_cost": 0.5, "backorder_cost": 1.0}
EVAL_SEED_BASE = 100000
HELDOUT_LAMBDAS = [6.0, 10.0, 14.0, 18.0, 22.0]
MIN_SEEDS = 10
TOST_BAND_FRAC = 0.02          # +/-2% of the no-comm cost (preregistered equivalence band)


# ----------------------------------------------------------------------- loaders
def load_bw_dir(draco_dir):
    """seed{S}_bw.json -> {seed: {lambda: {echelon: BW_cum}}}."""
    out = {}
    for p in sorted(glob.glob(os.path.join(draco_dir, "seed*_bw.json"))):
        base = os.path.basename(p)
        try:
            s = int("".join(ch for ch in base.split("seed")[1].split("_")[0] if ch.isdigit()))
        except (IndexError, ValueError):
            continue
        with open(p) as f:
            out[s] = {float(k): {a: float(b) for a, b in v.items()} for k, v in json.load(f).items()}
    return out


# ----------------------------------------------------------------------- bullwhip
def _roll_orders(policy, env, seed):
    """Torch-free roll of a baselines policy (act(obs, env) -> {agent:[frac]}); returns per-echelon
    BW_cum = Var(orders)/Var(customer demand) and total cost."""
    obs, _ = env.reset(seed=seed)
    policy.reset()
    orders = {a: [] for a in AGENTS}
    cust, cost = [], 0.0
    while True:
        acts = policy.act(obs, env)
        for a in AGENTS:
            orders[a].append(int(np.floor(np.clip(acts[a][0], 0, 1) * env.max_order + 0.5)))
        obs, _, _, trunc, info = env.step(acts)
        cust.append(float(env.current_incoming_order["retailer"]))
        cost += sum(info[a]["local_cost"] for a in AGENTS)
        if any(trunc.values()):
            break
    cv = float(np.var(cust))
    bw = {a: (float(np.var(orders[a]) / cv) if cv > 1e-9 else float("nan")) for a in AGENTS}
    return bw, cost


def baseline_bullwhip(make_pol, lambdas, episodes, env_cfg):
    Lam = _make_lambda_env_class(BeerGameParallelEnv)
    bw = {a: [] for a in AGENTS}
    costs = []
    for lam in lambdas:
        for e in range(episodes):
            env = Lam({**env_cfg, "demand_type": "poisson", "poisson_lam": float(lam)})
            b, c = _roll_orders(make_pol(), env, EVAL_SEED_BASE + e)
            for a in AGENTS:
                bw[a].append(b[a])
            costs.append(c)
    return {a: float(np.nanmean(bw[a])) for a in AGENTS}, float(np.mean(costs))


def draco_bullwhip(bw_by_seed, lambdas):
    lset = set(float(l) for l in lambdas)
    per_ech = {a: [] for a in AGENTS}
    for d in bw_by_seed.values():
        for a in AGENTS:
            per_ech[a].extend([d[l][a] for l in d if float(l) in lset])
    return {a: float(np.nanmean(per_ech[a])) if per_ech[a] else float("nan") for a in AGENTS}


# ----------------------------------------------------------------------- prereg check
def prereg_check(draco, lambdas, min_seeds=MIN_SEEDS):
    msgs = []
    n = len(draco)
    if n < min_seeds:
        msgs.append(f"WARNING: {n} seeds < preregistered minimum {min_seeds} (pilot only; CIs wide).")
    have = sorted({float(l) for d in draco.values() for l in d})
    want = sorted(float(l) for l in lambdas)
    if have != want:
        msgs.append(f"WARNING: dump lambdas {have} != preregistered {want}.")
    return msgs


# ----------------------------------------------------------------------- communication
def comm_analysis(comm_dirs, band_frac=TOST_BAND_FRAC, alpha=0.05):
    """comm_dirs[0] = no-comm reference dir; the rest = topology arms. Each dir has per-seed
    seed{S}.json ({lambda: cost}). Per topology: CRN-paired diff (no_comm - topology) over seeds,
    Wilcoxon (effect) + TOST (equivalence vs +/- band) ; Holm across topologies."""
    base = load_draco_dir(comm_dirs[0])
    base_seed_cost = {s: float(np.mean(list(d.values()))) for s, d in base.items()}
    band = band_frac * float(np.mean(list(base_seed_cost.values())))
    rows, wilcox_p = {}, {}
    for cd in comm_dirs[1:]:
        arm = load_draco_dir(cd)
        seeds = sorted(set(base_seed_cost) & set(arm))
        diffs = np.array([base_seed_cost[s] - float(np.mean(list(arm[s].values()))) for s in seeds])
        name = os.path.basename(os.path.normpath(cd))
        pr = paired(diffs, np.zeros_like(diffs))          # H0: no comm value
        eq = tost(diffs, -band, band, alpha=alpha)        # equivalence within +/- band
        rows[name] = {"n": len(seeds), "mean_value": float(diffs.mean()),
                      "ci": bootstrap_ci(diffs), "wilcoxon_p": pr["wilcoxon_p"],
                      "tost_p": eq["p_tost"], "equivalent": eq["equivalent"], "band": band}
        wilcox_p[name] = pr["wilcoxon_p"]
    holm = compare_many({k: v for k, v in wilcox_p.items() if v == v}, method="holm")  # drop NaN p's
    for name in rows:
        rows[name]["wilcoxon_p_holm"] = holm.get(name, {}).get("adjusted")
    return {"band": band, "topologies": rows}


# ----------------------------------------------------------------------- report
def run(draco_dir, refs_json, lambdas=HELDOUT_LAMBDAS, bullwhip_episodes=100, comm_dirs=None,
        out_path="results/confirmatory_report.json", min_seeds=MIN_SEEDS, verbose=True):
    draco = load_draco_dir(draco_dir)
    bw_by_seed = load_bw_dir(draco_dir)
    rungs = load_rungs(refs_json)
    raw = json.load(open(refs_json))
    bar_levels = raw.get("meta", {}).get("bar_levels")

    warnings = prereg_check(draco, lambdas, min_seeds)
    rep = summarize(draco, rungs, lambdas=lambdas)
    if verbose:
        for w in warnings:
            print(f"  [prereg] {w}")
        print_report(rep)

    # ---- bullwhip table (DRACO from dumps; static-BAR + Bayes computed offline) ----
    bull = {}
    if bw_by_seed:
        bull["DRACO"] = draco_bullwhip(bw_by_seed, lambdas)
    if bar_levels is not None:
        bw, c = baseline_bullwhip(lambda: BaseStockPolicy(bar_levels), lambdas, bullwhip_episodes, ENV_BASE)
        bull["static-BAR"] = dict(bw, cost=c)
    bw, c = baseline_bullwhip(lambda: make_bayes_rung(h=ENV_BASE["holding_cost"], b=ENV_BASE["backorder_cost"]),
                              lambdas, bullwhip_episodes, ENV_BASE)
    bull["Bayes-floor"] = dict(bw, cost=c)
    if verbose:
        print("-" * 78)
        print("  BULLWHIP (BW_cum = Var(orders)/Var(customer demand); the 'adapt without bullwhip' test)")
        print(f"    {'policy':<14}" + "".join(f"{('BW_'+a[:4]):>9}" for a in AGENTS))
        for name, d in bull.items():
            print(f"    {name:<14}" + "".join(f"{d.get(a, float('nan')):>9.2f}" for a in AGENTS))
        print("    (static-BAR ~1 = bullwhip-free; Bayes floor amplifies; DRACO should stay low while "
              "matching the regime.)")

    comm = comm_analysis(comm_dirs) if comm_dirs and len(comm_dirs) >= 2 else None
    if verbose and comm:
        print("-" * 78)
        print(f"  COMMUNICATION value (CRN-paired vs no-comm; TOST band +/-{comm['band']:.1f}; Holm-adjusted)")
        print(f"    {'topology':<18}{'value':>10}{'wilcoxon':>10}{'holm':>9}{'TOST':>8}{'equiv?':>8}")
        for name, r in comm["topologies"].items():
            print(f"    {name:<18}{r['mean_value']:>10.1f}{r['wilcoxon_p']:>10.3g}"
                  f"{(r['wilcoxon_p_holm'] if r['wilcoxon_p_holm'] is not None else float('nan')):>9.3g}"
                  f"{r['tost_p']:>8.3g}{str(r['equivalent']):>8}")

    out = {"warnings": warnings, "c1": rep["aggregate"], "per_lambda": rep["per_lambda"],
           "bullwhip": bull, "communication": comm, "n_seeds": rep["n_seeds"], "lambdas": rep["lambdas"]}
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=float)
    if verbose:
        print(f"\n  -> wrote {out_path}")
    return out


# ----------------------------------------------------------------------- self-test
def _selftest():
    import tempfile
    print("=" * 70)
    print("SELF-TEST: run_confirmatory_report on synthetic dumps (torch-free; uses the env)")
    print("=" * 70)
    lams = [8.0, 16.0]
    tmp = tempfile.mkdtemp()
    ddir = os.path.join(tmp, "draco_c1"); os.makedirs(ddir, exist_ok=True)
    rng = np.random.default_rng(0)
    for s in range(10, 13):                                   # 3 seeds (will trip the >=10 warning)
        cost = {str(l): 2400 + 30 * (l - 12) + rng.normal(0, 40) for l in lams}
        bw = {str(l): {a: 1.0 + i + rng.normal(0, 0.1) for i, a in enumerate(AGENTS)} for l in lams}
        json.dump(cost, open(os.path.join(ddir, f"seed{s}.json"), "w"))
        json.dump(bw, open(os.path.join(ddir, f"seed{s}_bw.json"), "w"))
    # refs JSON in the baselines schema, with meta.bar_levels for the static row
    refs = {"rungs": {"BAR_static": {str(l): 4700.0 + 60 * (l - 14) for l in lams},
                      "Bayes": {str(l): 7000.0 for l in lams},
                      "Oracle": {str(l): 2100.0 + 30 * (l - 14) for l in lams}},
            "lambdas": lams, "meta": {"bar_levels": [92, 92, 92, 92]}}
    rpath = os.path.join(tmp, "refs.json"); json.dump(refs, open(rpath, "w"))
    out = run(ddir, rpath, lambdas=lams, bullwhip_episodes=4,
              out_path=os.path.join(tmp, "confirmatory_report.json"), verbose=True)
    assert "gap_mean" in out["c1"] and out["n_seeds"] == 3
    assert "DRACO" in out["bullwhip"] and "static-BAR" in out["bullwhip"] and "Bayes-floor" in out["bullwhip"]
    assert any("seeds <" in w for w in out["warnings"])      # the >=10 prereg warning fired
    # comm self-test: two arms (no-comm dir + a cheaper "full" dir)
    cdir0 = os.path.join(tmp, "c2_nocomm"); cdir1 = os.path.join(tmp, "c2_full")
    os.makedirs(cdir0, exist_ok=True); os.makedirs(cdir1, exist_ok=True)
    for s in range(10, 15):
        json.dump({"8.0": 3000.0 + rng.normal(0, 20)}, open(os.path.join(cdir0, f"seed{s}.json"), "w"))
        json.dump({"8.0": 3005.0 + rng.normal(0, 20)}, open(os.path.join(cdir1, f"seed{s}.json"), "w"))
    comm = comm_analysis([cdir0, cdir1])
    assert "c2_full" in comm["topologies"] and comm["topologies"]["c2_full"]["n"] == 5
    print("\nrun_confirmatory_report self-test PASS")


def main():
    if len(sys.argv) > 1:
        ap = argparse.ArgumentParser()
        ap.add_argument("--draco-dir", required=True)
        ap.add_argument("--refs", required=True)
        ap.add_argument("--lambdas", nargs="+", type=float, default=HELDOUT_LAMBDAS)
        ap.add_argument("--bullwhip-episodes", type=int, default=100)
        ap.add_argument("--comm", nargs="+", default=None,
                        help="dirs: first = no-comm reference, rest = topology arms")
        ap.add_argument("--out", default="results/confirmatory_report.json")
        ap.add_argument("--min-seeds", type=int, default=MIN_SEEDS)
        a = ap.parse_args()
        run(a.draco_dir, a.refs, lambdas=a.lambdas, bullwhip_episodes=a.bullwhip_episodes,
            comm_dirs=a.comm, out_path=a.out, min_seeds=a.min_seeds)
    else:
        _selftest()


if __name__ == "__main__":
    main()
