"""
c1_stats.py -- statistical inference for the C1 (regime-inference) result.

Turns the headline DRACO number from a point estimate ("Gap_Recovered crossed 0") into a
defended claim ("Gap CI excludes 0; DRACO is significantly below the analytic Bayes-adaptive
policy"). The unit of analysis is the INDEPENDENT TRAINING SEED: each seed = one trained
policy, scored deterministically per-lambda on the EVAL seeds (so DRACO and the reference
rungs are CRN-comparable).

It provides:
  * load_rungs / mean_refs   -- read results/baselines_regime_v2.json (the 4-rung ladder from
                                `python scripts/baselines.py regime`). Used by the trainer and
                                eval as the single source of truth for BAR / Oracle / Bayes.
  * summarize / print_report -- aggregate Gap_Recovered mean [95% CI], the "CI excludes 0" flag,
                                the Gap<=1 oracle-validity flag (T3.3), per-lambda Gap curves,
                                rliable IQM + performance profile (Agarwal et al. 2021), and the
                                paired DRACO-vs-Bayes Wilcoxon / sign test.
  * paired / performance_profile / iqm / bootstrap_ci -- the building blocks (reusable for the
                                Phase-3 comm TOST in phase3_sweep.sh).

DATA MODEL
  draco : {seed(int): {lambda(float): mean_cost(float)}}   # one per-lambda vector per seed,
          e.g. results/draco_c1/seed{S}.json  (see eval_draco_v4.py --dump-c1)
  rungs : {name: {lambda(float): cost(float)}}             # from load_rungs(...)

Gap_Recovered (aggregate, matches heldout_eval.py): with BAR/Oracle the means over lambda,
    gap(seed) = (BAR - mean_lambda DRACO[seed]) / (BAR - Oracle)
  0 = matches the deployable static base-stock, 1 = matches the per-lambda oracle, >1 = beats
  the static oracle (finite-horizon effect; flags the oracle as not a valid static bound).

Run:
  python scripts/c1_stats.py                                   # self-test (numpy only)
  python scripts/c1_stats.py report --draco-dir results/draco_c1 \
                                    --refs results/baselines_regime_v2.json
"""
import os
import sys
import json
import glob
import numpy as np


# ==============================================================================
# Reference loader (single source of truth: baselines_regime_v2.json)
# ==============================================================================
def load_rungs(path):
    """Load the four-rung ladder written by baselines.regime_benchmark. Returns
    {name: {lambda(float): cost(float)}} for the rungs (BAR_static / Adaptive / Bayes / Oracle).
    JSON object keys are strings -> coerced back to float lambdas here."""
    with open(path) as f:
        data = json.load(f)
    rungs = data.get("rungs", data)                 # tolerate a bare {rung: {...}} too
    out = {}
    for name, d in rungs.items():
        if isinstance(d, dict):
            out[name] = {float(k): float(v) for k, v in d.items()}
    return out


def mean_refs(rungs, lambdas=None):
    """Mean cost per rung over `lambdas` (intersected with what each rung actually has, so a
    validation-lambda run falls back cleanly when the JSON only holds the test lambdas).
    Returns {name: mean_cost} for rungs that have at least one of the requested lambdas."""
    out = {}
    for name, d in rungs.items():
        lams = sorted(d) if lambdas is None else [float(l) for l in lambdas if float(l) in d]
        if lams:
            out[name] = float(np.mean([d[l] for l in lams]))
    return out


# ==============================================================================
# Statistical primitives
# ==============================================================================
def bootstrap_ci(values, stat=np.mean, n_boot=10000, ci=0.95, seed=0):
    """Percentile bootstrap CI of `stat` over the sample (resampling the seeds)."""
    v = np.asarray(values, float)
    n = v.size
    if n == 0:
        return (float("nan"), float("nan"))
    if n == 1:
        s = float(stat(v))
        return (s, s)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    boots = np.array([float(stat(v[i])) for i in idx])
    lo = float(np.percentile(boots, 100.0 * (1.0 - ci) / 2.0))
    hi = float(np.percentile(boots, 100.0 * (1.0 + ci) / 2.0))
    return (lo, hi)


def iqm(values):
    """Interquartile mean (rliable): mean of the middle 50% (drops the lowest/highest 25%).
    Robust to outlier seeds; the recommended central tendency for small-n RL (Agarwal 2021)."""
    v = np.sort(np.asarray(values, float))
    n = v.size
    if n == 0:
        return float("nan")
    k = int(np.floor(n * 0.25))
    core = v[k:n - k] if n - 2 * k > 0 else v
    return float(np.mean(core))


def performance_profile(scores, taus=None):
    """rliable run-score distribution: fraction of seeds with score >= tau, over a tau grid.
    Use oracle-normalized scores (0=BAR, 1=Oracle) so the profile is comparable across studies."""
    s = np.asarray(scores, float)
    if taus is None:
        lo = float(min(0.0, np.nanmin(s))) - 0.1 if s.size else 0.0
        hi = float(max(1.0, np.nanmax(s))) + 0.1 if s.size else 1.0
        taus = np.linspace(lo, hi, 101)
    taus = np.asarray(taus, float)
    frac = np.array([float(np.mean(s >= t)) for t in taus]) if s.size else np.zeros_like(taus)
    return {"tau": taus, "frac": frac}


def paired(a, b, alternative="two-sided"):
    """Paired nonparametric comparison of two equal-length samples (a vs b). Reports the mean
    difference (a-b), the Wilcoxon signed-rank p, and the sign-test (binomial) p.

    NOTE on small n: with 5 seeds, a perfectly consistent effect still gives Wilcoxon/sign
    p ~= 0.06 (you cannot cross 0.05 nonparametrically at n=5). Read the bootstrap CI as the
    primary inferential tool at small n, and scale seeds for the final claim."""
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    d = a - b
    n = int(d.size)
    out = {"n": n, "mean_diff": float(np.mean(d)) if n else float("nan"),
           "wilcoxon_p": float("nan"), "sign_p": float("nan"), "n_pos": 0, "n_nonzero": 0}
    nz = d[d != 0.0]
    out["n_nonzero"] = int(nz.size)
    out["n_pos"] = int(np.sum(nz > 0))
    if nz.size >= 1:
        try:
            from scipy import stats
            try:
                out["wilcoxon_p"] = float(stats.wilcoxon(nz, alternative=alternative).pvalue)
            except Exception:
                out["wilcoxon_p"] = float("nan")
            try:
                out["sign_p"] = float(stats.binomtest(out["n_pos"], nz.size, 0.5,
                                                      alternative=alternative).pvalue)
            except AttributeError:                       # scipy < 1.7
                out["sign_p"] = float(stats.binom_test(out["n_pos"], nz.size, 0.5,
                                                       alternative=alternative))
            except Exception:
                out["sign_p"] = float("nan")
        except Exception:
            pass                                         # scipy missing -> leave p's as nan
    return out


# ==============================================================================
# The C1 report
# ==============================================================================
def _common_lambdas(draco, rungs, keys, lambdas):
    sets = [set(draco[s]) for s in draco] + [set(rungs[k]) for k in keys if k in rungs]
    common = set.intersection(*sets) if sets else set()
    if lambdas is not None:
        common &= {float(l) for l in lambdas}
    return sorted(common)


def summarize(draco, rungs, bar_key="BAR_static", ceiling_key="Oracle", bayes_key="Bayes",
              lambdas=None, n_boot=10000, ci=0.95, gap_tol=1e-6):
    """Full inferential summary of the C1 result. See module docstring for the data model."""
    draco = {int(s): {float(k): float(v) for k, v in d.items()} for s, d in draco.items()}
    seeds = sorted(draco)
    lams = _common_lambdas(draco, rungs, [bar_key, ceiling_key], lambdas)
    if not seeds or not lams:
        raise ValueError(f"no overlapping seeds/lambdas (seeds={seeds}, lambdas={lams})")

    bar_l = {l: rungs[bar_key][l] for l in lams}
    orc_l = {l: rungs[ceiling_key][l] for l in lams}
    has_bayes = bayes_key in rungs and all(l in rungs[bayes_key] for l in lams)
    bay_l = {l: rungs[bayes_key][l] for l in lams} if has_bayes else None

    bar = float(np.mean([bar_l[l] for l in lams]))
    orc = float(np.mean([orc_l[l] for l in lams]))
    bay = float(np.mean([bay_l[l] for l in lams])) if has_bayes else None
    denom = max(1e-9, bar - orc)

    # per-seed aggregate cost + gap (oracle-normalized score)
    D = np.array([float(np.mean([draco[s][l] for l in lams])) for s in seeds])
    gap = (bar - D) / denom

    # per-lambda per-seed gaps (for the Gap<=1 validity check and the per-lambda curve)
    gap_le1_all = True
    per_lambda = {}
    for l in lams:
        d_l = np.array([draco[s][l] for s in seeds])
        denom_l = max(1e-9, bar_l[l] - orc_l[l])
        g_l = (bar_l[l] - d_l) / denom_l
        le1 = bool(np.all(g_l <= 1.0 + gap_tol))
        gap_le1_all = gap_le1_all and le1
        rec = {"draco_mean": float(d_l.mean()),
               "draco_ci": bootstrap_ci(d_l, n_boot=n_boot, ci=ci),
               "bar": bar_l[l], "oracle": orc_l[l],
               "gap_mean": float(g_l.mean()),
               "gap_ci": bootstrap_ci(g_l, n_boot=n_boot, ci=ci),
               "gap_le1": le1}
        if has_bayes:
            rec["bayes"] = bay_l[l]
            rec["draco_minus_bayes"] = float(d_l.mean() - bay_l[l])
        per_lambda[l] = rec

    gap_ci = bootstrap_ci(gap, n_boot=n_boot, ci=ci)
    iqm_ci = bootstrap_ci(gap, stat=iqm, n_boot=n_boot, ci=ci)
    agg = {
        "draco_cost_mean": float(D.mean()),
        "draco_cost_ci": bootstrap_ci(D, n_boot=n_boot, ci=ci),
        "bar": bar, "oracle": orc, "bayes": bay,
        "gap_mean": float(gap.mean()),
        "gap_ci": gap_ci,
        "gap_ci_excludes_0": bool(gap_ci[0] > 0.0),       # the inferential "crossed 0"
        "iqm_gap": iqm(gap),
        "iqm_gap_ci": iqm_ci,
        "gap_le1_check": bool(gap_le1_all),               # T3.3: is the static oracle a valid bound?
    }

    if has_bayes:
        # paired one-sample test over seeds: is DRACO below the (fixed) Bayes-adaptive cost?
        # diff = Bayes - DRACO_cost(seed)  ->  positive = DRACO cheaper.
        vs = paired(np.full_like(D, bay), D, alternative="two-sided")
        vs_ci = bootstrap_ci(bay - D, n_boot=n_boot, ci=ci)
        vs["diff_ci"] = vs_ci
        vs["bayes_gap_mean"] = float(np.mean((bay - D) / max(1e-9, bay - orc)))
        vs["beats_bayes_ci"] = bool(vs_ci[0] > 0.0)       # CI of (Bayes - DRACO) excludes 0
        agg["vs_bayes"] = vs

    return {
        "n_seeds": len(seeds), "seeds": seeds, "lambdas": lams,
        "aggregate": agg, "per_lambda": per_lambda,
        "performance_profile": performance_profile(gap),
        "has_bayes": has_bayes,
    }


def print_report(rep):
    a = rep["aggregate"]
    print("=" * 78)
    print(f"C1 STATISTICAL REPORT  |  {rep['n_seeds']} seeds  |  lambdas "
          f"{[f'{l:g}' for l in rep['lambdas']]}")
    print("=" * 78)

    def _ci(t):
        return f"[{t[0]:.3f}, {t[1]:.3f}]"

    def _ci1(t):
        return f"[{t[0]:.1f}, {t[1]:.1f}]"

    print(f"  DRACO mean cost     : {a['draco_cost_mean']:.1f}  95% CI {_ci1(a['draco_cost_ci'])}")
    print(f"  references          : BAR(static)={a['bar']:.1f}   Oracle={a['oracle']:.1f}"
          + (f"   Bayes={a['bayes']:.1f}" if a['bayes'] is not None else "   Bayes=n/a"))
    print("-" * 78)
    flag = "PASS (excludes 0)" if a["gap_ci_excludes_0"] else "NOT SIGNIFICANT (CI includes 0)"
    print(f"  Gap_Recovered       : {a['gap_mean']:+.3f}  95% CI {_ci(a['gap_ci'])}   -> {flag}")
    print(f"  Gap_Recovered (IQM) : {a['iqm_gap']:+.3f}  95% CI {_ci(a['iqm_gap_ci'])}")
    le1 = "OK (oracle bounds DRACO)" if a["gap_le1_check"] else \
          "VIOLATED -> DRACO beats the STATIC oracle somewhere (finite-horizon effect; T3.3)"
    print(f"  Gap <= 1 check      : {le1}")

    if "vs_bayes" in a:
        vb = a["vs_bayes"]
        verdict = "DRACO significantly below Bayes" if vb["beats_bayes_ci"] else \
                  "not significant (CI includes 0)"
        print("-" * 78)
        print(f"  DRACO vs Bayes-adaptive (paired over seeds, +=DRACO cheaper):")
        print(f"    mean(Bayes - DRACO) = {vb['mean_diff']:+.1f}   95% CI {_ci1(vb['diff_ci'])}  -> {verdict}")
        print(f"    Bayes-anchored gap  = {vb['bayes_gap_mean']:+.3f}   "
              f"(1 = matches Bayes-vs-oracle headroom fully)")
        print(f"    Wilcoxon p = {vb['wilcoxon_p']:.4g}   sign-test p = {vb['sign_p']:.4g}   "
              f"({vb['n_pos']}/{vb['n_nonzero']} seeds cheaper)")
        if rep["n_seeds"] <= 6:
            print(f"    [note] n={rep['n_seeds']}: nonparametric p is floored ~0.06; the bootstrap CI is "
                  f"the primary test. Scale seeds for the final claim.")

    print("-" * 78)
    print(f"  per-lambda:")
    hdr = f"    {'lambda':>7}{'DRACO':>10}{'95% CI':>17}{'BAR':>8}"
    hdr += f"{'Bayes':>8}" if rep["has_bayes"] else ""
    hdr += f"{'Oracle':>8}{'Gap':>8}{'Gap CI':>16}{'<=1':>5}"
    print(hdr)
    for l in rep["lambdas"]:
        r = rep["per_lambda"][l]
        row = f"    {l:>7g}{r['draco_mean']:>10.1f}{_ci1(r['draco_ci']):>17}{r['bar']:>8.0f}"
        if rep["has_bayes"]:
            row += f"{r['bayes']:>8.0f}"
        row += f"{r['oracle']:>8.0f}{r['gap_mean']:>8.2f}{_ci(r['gap_ci']):>16}"
        row += f"{'y' if r['gap_le1'] else 'N':>5}"
        print(row)
    print("=" * 78)
    print("  (performance profile arrays are in rep['performance_profile'] for plotting:")
    print("   x=rep['performance_profile']['tau'], y=fraction of seeds with score>=tau.)")


def load_draco_dir(draco_dir):
    """Load results/draco_c1/seed{S}.json files -> {seed: {lambda: cost}}."""
    out = {}
    for p in sorted(glob.glob(os.path.join(draco_dir, "seed*.json"))):
        base = os.path.basename(p)
        try:
            s = int("".join(ch for ch in base.split("seed")[1] if ch.isdigit()))
        except (IndexError, ValueError):
            continue
        with open(p) as f:
            out[s] = {float(k): float(v) for k, v in json.load(f).items()}
    return out


# ==============================================================================
# CLI
# ==============================================================================
def _report_cli(argv):
    import argparse
    ap = argparse.ArgumentParser(prog="c1_stats.py report")
    ap.add_argument("--draco-dir", default="results/draco_c1",
                    help="dir of seed{S}.json (per-seed {lambda: cost}); see eval_draco_v4.py --dump-c1")
    ap.add_argument("--refs", default="results/baselines_regime_v2.json")
    ap.add_argument("--lambdas", nargs="+", type=float, default=None)
    args = ap.parse_args(argv)
    draco = load_draco_dir(args.draco_dir)
    if not draco:
        print(f"no seed*.json found in {args.draco_dir}. Generate them with eval_draco_v4.py --dump-c1.")
        return
    rungs = load_rungs(args.refs)
    print_report(summarize(draco, rungs, lambdas=args.lambdas))


def _selftest():
    print("=" * 70)
    print("SELF-TEST: c1_stats metric math (numpy only; scipy optional for p-values)")
    print("=" * 70)
    lams = [6.0, 10.0, 14.0, 18.0, 22.0]
    rungs = {
        "BAR_static": {l: 4000.0 for l in lams},
        "Oracle":     {l: 2000.0 for l in lams},
        "Bayes":      {l: 2500.0 for l in lams},
    }

    def draco_at(cost_fn):
        return {s: {l: float(cost_fn(s, l)) for l in lams} for s in range(5)}

    # Case A: DRACO == Oracle -> Gap == 1, oracle bound valid.
    rep = summarize(draco_at(lambda s, l: 2000.0), rungs, n_boot=2000)
    assert abs(rep["aggregate"]["gap_mean"] - 1.0) < 1e-6, rep["aggregate"]["gap_mean"]
    assert rep["aggregate"]["gap_le1_check"] is True
    print(f"  A) DRACO=Oracle      -> gap={rep['aggregate']['gap_mean']:.3f} (==1), Gap<=1 OK  PASS")

    # Case B: DRACO == BAR -> Gap == 0, CI must NOT exclude 0.
    rep = summarize(draco_at(lambda s, l: 4000.0), rungs, n_boot=2000)
    assert abs(rep["aggregate"]["gap_mean"]) < 1e-6
    assert rep["aggregate"]["gap_ci_excludes_0"] is False
    print(f"  B) DRACO=BAR         -> gap={rep['aggregate']['gap_mean']:.3f} (==0), CI!>0  PASS")

    # Case C: DRACO between, with per-seed noise -> 0<gap<1, CI brackets the mean, beats Bayes.
    rng = np.random.default_rng(1)
    noise = {s: rng.normal(0, 60) for s in range(5)}
    rep = summarize(draco_at(lambda s, l: 3000.0 + noise[s]), rungs, n_boot=4000)
    g = rep["aggregate"]["gap_mean"]; lo, hi = rep["aggregate"]["gap_ci"]
    assert 0.0 < g < 1.0 and lo <= g <= hi
    assert rep["has_bayes"] and rep["aggregate"]["vs_bayes"]["mean_diff"] < 0  # Bayes(2500)-DRACO(3000)<0
    print(f"  C) DRACO=3000(+noise)-> gap={g:.3f} in (0,1), CI=[{lo:.3f},{hi:.3f}], "
          f"vs-Bayes mean_diff={rep['aggregate']['vs_bayes']['mean_diff']:+.0f} (Bayes cheaper)  PASS")

    # Case D: DRACO below the static oracle -> Gap>1, validity flag must trip; DRACO beats Bayes.
    rep = summarize(draco_at(lambda s, l: 1500.0 + 20 * (s - 2)), rungs, n_boot=2000)
    assert rep["aggregate"]["gap_mean"] > 1.0
    assert rep["aggregate"]["gap_le1_check"] is False
    vb = rep["aggregate"]["vs_bayes"]
    assert vb["mean_diff"] > 0 and vb["diff_ci"][0] > 0  # Bayes(2500)-DRACO(~1500)>0, CI excludes 0
    print(f"  D) DRACO=1500        -> gap={rep['aggregate']['gap_mean']:.3f} (>1), Gap<=1 VIOLATED "
          f"(detected), beats-Bayes CI excludes 0  PASS")

    # primitives
    assert abs(iqm([1, 2, 3, 4, 5]) - 3.0) < 1e-9                  # middle-3 mean of 1..5
    assert abs(iqm([1, 2, 3, 4, 5, 6, 100, 200]) - 4.5) < 1e-9     # middle-4 [3,4,5,6]; outliers dropped
    pp = performance_profile([0.0, 0.5, 1.0], taus=[0.0, 0.6, 1.1])
    assert pp["frac"][0] == 1.0 and abs(pp["frac"][1] - (1 / 3)) < 1e-9 and pp["frac"][2] == 0.0
    print("  E) iqm + performance_profile primitives  PASS")
    print("\nc1_stats self-test PASS")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "report":
        _report_cli(sys.argv[2:])
    else:
        _selftest()
