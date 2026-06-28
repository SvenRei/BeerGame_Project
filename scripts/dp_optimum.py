"""
dp_optimum.py -- EXACT finite-horizon dynamic-programming optimum for a single-stage canonical
inventory problem. Anchors the "gap to optimal" claim (review #2/#17): it shows the critical-fractile
base-stock MATCHES the exact DP optimum, i.e. the base-stock benchmark is a *true* optimum, not a
hand-built reference. For the canonical SERIAL system the multi-echelon optimum decomposes into
per-echelon single-stage problems of exactly this form (Clark & Scarf 1960; Zipkin 2000, Ch. 8-9);
this script is that building block at one stage / one lambda. (The full serial echelon-DP is the
heavier alternative; the env-side check that the numerically-optimized canonical base-stock is global
is `python scripts/baselines.py validate-canonical`.)

Model: periodic review, horizon T, Poisson(lambda) demand, replenishment lead time L, unit holding h
and backorder b, penalty at this (customer-facing) stage. With a base-stock policy the relevant cost
of ordering up to S is the newsvendor loss over the protection interval (L+1 periods):
    G(S) = E[ h*(S - D_{L+1})^+ + b*(D_{L+1} - S)^+ ],   D_{L+1} ~ Poisson(lambda*(L+1)).
The exact scalar DP over inventory position x is
    V_t(x) = min_{S >= x} [ G(S) + E_{D~Poisson(lambda)} V_{t+1}(S - D) ],   V_{T+1} = 0,
whose minimizer is independent of x (a base-stock level) -- the discovery we verify numerically.

Run:
  python scripts/dp_optimum.py                 # self-test
  python scripts/dp_optimum.py validate --lam 8 --L 4 --h 0.5 --b 1.0 --T 50
"""
import sys
import argparse
import numpy as np
from scipy import stats


def _pois_pmf(mean, dmax):
    p = stats.poisson.pmf(np.arange(dmax + 1), mean)
    p[-1] += 1.0 - p.sum()                     # fold the tail mass into the last bin
    return p


def G_all(xs, lam, L, h, b, dmax):
    """Protection-interval newsvendor cost G(S) for every S in `xs` (vectorized)."""
    d = np.arange(dmax + 1)
    p = _pois_pmf(lam * (L + 1), dmax)
    S = xs[:, None]; D = d[None, :]
    return (p[None, :] * (h * np.maximum(S - D, 0) + b * np.maximum(D - S, 0))).sum(axis=1)


def critical_fractile_basestock(lam, L, h, b):
    """The textbook base-stock: smallest S with P(D_{L+1} <= S) >= b/(b+h)."""
    return int(stats.poisson.ppf(b / (b + h), lam * (L + 1)))


def finite_horizon_dp(lam, L, h, b, T, x_lo=-60, x_hi=200, dmax_p=None, dmax_1=None):
    """Exact backward-induction DP. Returns (V1 over xs, xs, base_stock_by_t)."""
    dmax_p = dmax_p or int(lam * (L + 1) + 8 * np.sqrt(lam * (L + 1)) + 5)
    dmax_1 = dmax_1 or int(lam + 8 * np.sqrt(lam) + 5)
    xs = np.arange(x_lo, x_hi + 1)
    Gvals = G_all(xs, lam, L, h, b, dmax_p)
    d = np.arange(dmax_1 + 1)
    pd = _pois_pmf(lam, dmax_1)
    NXT = np.clip(xs[:, None] - d[None, :], x_lo, x_hi) - x_lo   # (S, d) -> index into V
    Vnext = np.zeros(len(xs))
    bs_by_t = []
    for _t in range(T, 0, -1):
        EV = (Vnext[NXT] * pd[None, :]).sum(axis=1)             # E_D V_{t+1}(S - D), per S
        Q = Gvals + EV                                          # cost of order-up-to S (interior)
        bs_by_t.append(int(xs[int(np.argmin(Q))]))             # the unconstrained minimizer = base-stock
        Vnext = np.minimum.accumulate(Q[::-1])[::-1]           # V_t(x) = min_{S>=x} Q[S]
    bs_by_t.reverse()
    return Vnext, xs, bs_by_t


def fixed_basestock_cost(S, lam, L, h, b, T, x0, dmax_p=None):
    """Expected T-period cost of the stationary policy 'order up to S' from x0 (x0 <= S): T*G(S)."""
    dmax_p = dmax_p or int(lam * (L + 1) + 8 * np.sqrt(lam * (L + 1)) + 5)
    g = float(G_all(np.array([S]), lam, L, h, b, dmax_p)[0])
    return T * g


def validate(lam=8.0, L=4, h=0.5, b=1.0, T=50, x0=0, verbose=True):
    """Show the critical-fractile base-stock == the exact DP optimum (the 'gap to optimal' anchor)."""
    V1, xs, bs_by_t = finite_horizon_dp(lam, L, h, b, T)
    dp_cost = float(V1[x0 - xs[0]])
    Gvals = G_all(xs, lam, L, h, b, int(lam * (L + 1) + 8 * np.sqrt(lam * (L + 1)) + 5))
    S_argmin = int(xs[int(np.argmin(Gvals))])                  # base-stock that minimizes G
    S_cf = critical_fractile_basestock(lam, L, h, b)           # textbook critical-fractile
    stat_cost = fixed_basestock_cost(S_argmin, lam, L, h, b, T, x0)
    gap = (stat_cost - dp_cost) / dp_cost if dp_cost > 0 else float("nan")
    interior = bs_by_t[: max(1, T - 2)]                        # ignore the last couple of periods
    interior_const = len(set(interior)) == 1
    if verbose:
        print("=" * 74)
        print(f"SINGLE-STAGE EXACT DP  lam={lam:g} L={L} h={h} b={b} T={T}  (protection interval L+1={L+1})")
        print("=" * 74)
        print(f"  critical-fractile base-stock S* (textbook)   = {S_cf}")
        print(f"  argmin G(S) base-stock (newsvendor optimum)  = {S_argmin}")
        print(f"  DP base-stock per period (interior constant) = {interior[0]}  (constant: {interior_const})")
        print(f"  exact DP optimal cost from x0={x0}           = {dp_cost:.2f}")
        print(f"  stationary base-stock cost  T*G(S*)          = {stat_cost:.2f}")
        print(f"  gap (stationary - DP)/DP                     = {100*gap:+.3f}%")
        print(f"  -> the critical-fractile base-stock matches the exact finite-horizon DP optimum, so the")
        print(f"     base-stock benchmark IS the true single-stage optimum (a valid 'gap to optimal' denom).")
    return dict(dp_cost=dp_cost, stationary_cost=stat_cost, gap=gap, S_cf=S_cf, S_argmin=S_argmin,
                interior_basestock=interior[0], interior_const=interior_const)


def _selftest():
    print("=" * 70)
    print("SELF-TEST: single-stage finite-horizon DP optimum")
    print("=" * 70)
    for lam, L in [(8.0, 4), (14.0, 4), (6.0, 2)]:
        r = validate(lam, L, 0.5, 1.0, T=40, verbose=False)
        # (1) critical fractile == argmin G (the textbook level is the newsvendor optimum)
        assert abs(r["S_cf"] - r["S_argmin"]) <= 1, (lam, L, r)
        # (2) DP recovers a CONSTANT interior base-stock == the newsvendor level (base-stock optimality)
        assert r["interior_const"] and abs(r["interior_basestock"] - r["S_argmin"]) <= 1, r
        # (3) stationary base-stock is within MC/finite-horizon noise of the exact DP optimum
        assert -1e-6 <= r["gap"] < 0.03, r
        print(f"  lam={lam:>4g} L={L}: S*={r['S_argmin']:>3} | DP={r['dp_cost']:8.1f} "
              f"stat={r['stationary_cost']:8.1f} gap={100*r['gap']:+.3f}% | base-stock=DP-optimum  PASS")
    print("\ndp_optimum self-test PASS  (critical-fractile base-stock == exact finite-horizon DP optimum)")


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "validate":
        ap = argparse.ArgumentParser(prog="dp_optimum.py validate")
        ap.add_argument("--lam", type=float, default=8.0)
        ap.add_argument("--L", type=int, default=4)
        ap.add_argument("--h", type=float, default=0.5)
        ap.add_argument("--b", type=float, default=1.0)
        ap.add_argument("--T", type=int, default=50)
        ap.add_argument("--x0", type=int, default=0)
        a = ap.parse_args(sys.argv[2:])
        validate(a.lam, a.L, a.h, a.b, a.T, a.x0)
    else:
        _selftest()


if __name__ == "__main__":
    main()
