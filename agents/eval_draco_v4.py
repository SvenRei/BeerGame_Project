"""
eval_draco_v4.py -- evaluation / benchmark loader for DRACO v4.

A v4 port of eval_draco_v3 (the v3 reference is left untouched). What it does:

  STANDARD BENCHMARK (per regime: poisson / black_swan / extreme_chaos)
    mean cost, CVaR cost, bullwhip, Type-1 (alpha) service, Type-2 (beta) fill,
    order jitter, and the zero-message comm value (paired Wilcoxon) -- identical
    formulas to the v3 protocol.

  REGIME-UNCERTAINTY (C1)
    runs the policy on the held-out stationary lambdas and reports per-lambda cost,
    mean cost, and Gap_Recovered vs the fixed-policy BAR / per-lambda CEILING from
    `python scripts/baselines.py regime`. This is the C1 number for a checkpoint.

  MESSAGE ANALYSIS (Study 2; only when the checkpoint has comm)
    per-channel message saturation, correlation of each channel with the SENDER's
    demand (does the message carry the demand signal?), and the see-through-bullwhip
    diagnostic: upstream demand-tracking error (|d_hat - true customer demand|) WITH
    vs WITHOUT messages. A positive delta = the shared signal helps upstream forecast
    (the Lee-Padmanabhan-Whang channel).

Topology: the ADJ is rebuilt from the checkpoint's `comm_topology` (via agents.topologies),
so a 'skip'/'full'-trained policy routes exactly as it did in training.

Usage:
  python agents/eval_draco_v4.py --ckpt weights_draco/run_dracov4_<id>/draco_checkpoint_best.pt
  python agents/eval_draco_v4.py --ckpt ... --episodes 100 --messages
  python agents/eval_draco_v4.py --ckpt ... --bar 4726 --ceiling 2202
"""
import os
import sys
import argparse
import numpy as np
import torch
from scipy import stats

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from envs.beer_game_env import BeerGameParallelEnv
from agents.draco_v4 import (
    make_encoder, make_actor, BaseStockActor, MessageHead, DemandRandomizedBeerGame,
)
from agents.topologies import get_adj

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
AGENTS = ["retailer", "wholesaler", "distributor", "manufacturer"]
SCENARIOS = ["poisson", "black_swan", "extreme_chaos"]
SEED_BASE = 2000
HELDOUT_SEED_BASE = 100000          # matches baselines.py SEED_BASE (CRN with the BAR/CEILING)
HELDOUT_LAMBDAS = [6.0, 10.0, 14.0, 18.0, 22.0]
ENV_BASE = {"horizon": 50, "max_order": 100}
H_COST = 0.5     # holding cost/unit/week -- MUST match the env/config (config.yaml: holding_cost)
B_COST = 1.0     # backorder cost/unit/week -- MUST match the env/config (config.yaml: backorder_cost)


def _safe_ratio(num, den):
    return float(num / den) if den and np.isfinite(den) and den != 0 else float("nan")


# ==============================================================================
# DRACO v4 policy (benchmark-protocol compatible; topology-aware)
# ==============================================================================
class DracoV4Policy:
    def __init__(self, ckpt, env, ablate=False, deterministic=True):
        self.env = env
        self.ablate = ablate
        self.deterministic = deterministic
        self.max_order = env.max_order
        self.N = len(AGENTS)
        cfg = ckpt.get("config", {}).get("agent", {})

        self.hidden = cfg.get("hidden_dim", 128)
        self.z_dim = cfg.get("z_dim", 8)
        self.msg_dim = cfg.get("msg_dim", 4)
        self.encoder_type = cfg.get("encoder_type", "gru")
        self.craft_max_len = max(int(cfg.get("craft_max_len", 64)), int(env.horizon) + 1)
        enc_cfg = dict(cfg); enc_cfg["craft_max_len"] = self.craft_max_len
        local_dim = env.observation_space("retailer").shape[0]
        self.use_context = cfg.get("use_context", True)

        # topology routed EXACTLY as the checkpoint was trained
        self.comm_topology = cfg.get("comm_topology", "neighbor")
        self.adj = get_adj(self.comm_topology).to(DEVICE)
        self.msg_mode = cfg.get("msg_mode", "learned")   # "learned" DIAL | "dhat" broadcast

        self.encoder = make_encoder(self.encoder_type, local_dim, self.msg_dim, self.z_dim, enc_cfg).to(DEVICE)
        self.encoder.load_state_dict(ckpt["encoder"]); self.encoder.eval()

        self.actor_head = cfg.get("actor_head", "structured")
        self.actors = []
        for sd in ckpt["actors"]:
            a = make_actor(self.actor_head, local_dim, self.z_dim, self.msg_dim,
                           self.hidden, self.max_order, cfg).to(DEVICE)
            a.load_state_dict(sd); a.eval()
            self.actors.append(a)
        self.has_dhat = hasattr(self.actors[0], "demand_estimate")
        self.use_comm = bool(cfg.get("use_comm", False))   # gate routing for no-comm checkpoints

        self.msg_heads = None
        if ckpt.get("msg_heads"):
            self.msg_heads = []
            for sd in ckpt["msg_heads"]:
                m = MessageHead(local_dim, self.z_dim, self.msg_dim, self.hidden).to(DEVICE)
                m.load_state_dict(sd); m.eval()
                self.msg_heads.append(m)

        self.reset()

    def reset(self):
        self.m_buf = torch.zeros(self.N, self.msg_dim, device=DEVICE)
        self._obs_hist = []
        self._msg_hist = []
        self.last_S = np.zeros(self.N)
        self.last_dhat = np.full(self.N, np.nan)
        self.last_msg = np.zeros((self.N, self.msg_dim))

    @torch.no_grad()
    def act(self, obs):
        o_t = torch.tensor(np.stack([obs[a] for a in AGENTS]), dtype=torch.float32, device=DEVICE)

        incoming = self.adj @ self.m_buf
        if self.ablate or not self.use_comm:
            incoming = torch.zeros_like(incoming)

        self._obs_hist.append(o_t)
        self._msg_hist.append(incoming)
        obs_seq = torch.stack(self._obs_hist[-self.craft_max_len:])
        msg_seq = torch.stack(self._msg_hist[-self.craft_max_len:])
        mu, _, _ = self.encoder.forward_sequence(obs_seq, msg_seq)
        z_t = mu[-1]
        z_act = z_t if self.use_context else torch.zeros_like(z_t)

        S = torch.zeros(self.N, 1, device=DEVICE)
        dhat = np.full(self.N, np.nan)
        for i in range(self.N):
            s_mu, s_std = self.actors[i](o_t[i:i + 1], z_act[i:i + 1], incoming[i:i + 1])
            S[i] = s_mu if self.deterministic else torch.distributions.Normal(s_mu, s_std).sample()
            if self.has_dhat:
                dhat[i] = float(self.actors[i].demand_estimate(z_act[i:i + 1]).reshape(-1)[0].item())

        m_out = torch.zeros(self.N, self.msg_dim, device=DEVICE)
        if self.use_comm and not self.ablate and (self.msg_mode == "dhat" or self.msg_heads is not None):
            if self.msg_mode == "dhat" and self.has_dhat:
                for i in range(self.N):
                    m_out[i] = self.actors[i].demand_estimate(z_act[i:i + 1]).reshape(-1) / 100.0
            else:
                for i in range(self.N):
                    m_out[i] = self.msg_heads[i](o_t[i:i + 1], z_t[i:i + 1]).squeeze(0)

        order, _ = BaseStockActor.order_from_S(S, o_t, self.max_order)
        frac = (order / self.max_order).clamp(0.0, 1.0)
        self.last_S = S.detach().cpu().numpy().reshape(-1)
        self.last_dhat = dhat
        self.last_msg = m_out.detach().cpu().numpy()
        # cache (obs, belief, incoming) so callers can re-evaluate an actor under a
        # COUNTERFACTUAL incoming message (causal positive-listening probe, Jaques et al. 2019).
        self.last_ctx = (o_t.detach(), z_act.detach(), incoming.detach())
        self.m_buf = m_out
        return {a: float(frac[i, 0].item()) for i, a in enumerate(AGENTS)}

    @torch.no_grad()
    def probe_S(self, agent_idx, msg_scaled):
        """Order-up-to S that actor `agent_idx` WOULD output at the last step's (obs, belief)
        if its incoming message were `msg_scaled` (in the /100 broadcast scale). Used to measure
        dS/d(broadcast demand) -- does the receiver causally respond to message CONTENT?"""
        o_t, z_act, _inc = self.last_ctx
        m = torch.full((1, self.msg_dim), float(msg_scaled), device=DEVICE)
        s_mu, _ = self.actors[agent_idx](o_t[agent_idx:agent_idx + 1], z_act[agent_idx:agent_idx + 1], m)
        return float(s_mu.reshape(-1)[0].item())


# ==============================================================================
# Episode rollout + metrics (v3 formulas; optional per-step trace for message analysis)
# ==============================================================================
def run_episode(policy, env, seed, trace=False):
    torch.manual_seed(seed)
    obs, _ = env.reset(seed=seed)
    policy.reset()
    orders = {a: [] for a in AGENTS}
    demand = {a: [] for a in AGENTS}
    inv = {a: [] for a in AGENTS}
    back = {a: [] for a in AGENTS}
    fill_d = {a: 0.0 for a in AGENTS}
    fill_m = {a: 0.0 for a in AGENTS}
    tot_cost = 0.0
    cust_series = []                                # customer demand each step (always collected)
    tr_dhat, tr_msg, tr_cust = [], [], []          # per-step traces (message analysis)
    while True:
        acts = policy.act(obs)
        for a in AGENTS:
            orders[a].append(int(np.floor(np.clip(acts[a], 0, 1) * env.max_order + 0.5)))
        cust = float(env.current_incoming_order["retailer"])   # true customer demand (~ regime)
        cust_series.append(cust)
        obs, _r, _t, truncs, infos = env.step({a: [acts[a]] for a in AGENTS})
        for a in AGENTS:
            tot_cost += infos[a]["local_cost"]
            inv[a].append(float(obs[a][0])); back[a].append(float(obs[a][1]))
            demand[a].append(float(env.current_incoming_order[a]))
            tt = infos[a].get("training_targets", {})
            fill_d[a] += tt.get("demand", infos[a].get("demand", 0.0))
            fill_m[a] += tt.get("demand_met", infos[a].get("demand_met", 0.0))
        if trace:
            tr_dhat.append(policy.last_dhat.copy())
            tr_msg.append(policy.last_msg.copy())
            tr_cust.append(cust)
        if any(truncs.values()):
            break
    all_inv = np.concatenate([inv[a] for a in AGENTS])
    all_back = np.concatenate([back[a] for a in AGENTS])
    cust_arr = np.asarray(cust_series, dtype=float)
    cust_sum = float(cust_arr.sum())

    # ---- per-stage supply-chain metrics (Chen et al. 2000; Lee et al. 1997; Disney-Towill 2003) ----
    cust_var = float(np.var(cust_arr))
    bw_stage, ovar, dvar, mean_inv, mean_back = {}, {}, {}, {}, {}
    serv_alpha, fill_beta_s, stockout_freq, hold_c, back_c = {}, {}, {}, {}, {}
    bw_cum, nsamp, mean_netstock, total_c = {}, {}, {}, {}
    for a in AGENTS:
        o = np.asarray(orders[a], float); d = np.asarray(demand[a], float)
        iv = np.asarray(inv[a], float); bk = np.asarray(back[a], float)
        ns = iv - bk                                              # net stock = on-hand - backlog
        ovar[a] = float(np.var(o)); dvar[a] = float(np.var(d))
        bw_stage[a] = _safe_ratio(ovar[a], dvar[a])               # LOCAL order-variance amplification
        bw_cum[a] = _safe_ratio(ovar[a], cust_var)                # CUMULATIVE bullwhip vs CUSTOMER demand (Lee/Chen)
        nsamp[a] = _safe_ratio(float(np.var(ns)), cust_var)       # Net Stock Amplification (Disney-Towill 2003)
        mean_inv[a] = float(iv.mean()); mean_back[a] = float(bk.mean()); mean_netstock[a] = float(ns.mean())
        serv_alpha[a] = float(np.mean(bk == 0))                   # Type-1 / cycle service level
        fill_beta_s[a] = _safe_ratio(fill_m[a], fill_d[a])        # Type-2 / fill rate
        stockout_freq[a] = float(np.mean(bk > 0))                 # P(stockout) per step
        hold_c[a] = H_COST * float(iv.sum())                     # holding-cost share
        back_c[a] = B_COST * float(bk.sum())                     # backorder-cost share
        total_c[a] = hold_c[a] + back_c[a]                       # total cost borne at this echelon
    hold_total = float(sum(hold_c.values())); back_total = float(sum(back_c.values()))

    out = {
        "cost": tot_cost,
        "cust_sum": cust_sum,
        "cost_per_demand": _safe_ratio(tot_cost, cust_sum),       # normalized cost
        "bw_overall": _safe_ratio(np.var(orders["manufacturer"]), np.var(demand["retailer"])),
        "bw_slope": float(np.polyfit(np.arange(len(AGENTS)),
                                     np.log([max(ovar[a], 1e-9) for a in AGENTS]), 1)[0]),  # variance amplification across chain
        "avg_inv": float(all_inv.mean()),
        "avg_back": float(all_back.mean()),
        "ret_service_alpha": float(np.mean(np.array(back["retailer"]) == 0)),
        "fill_beta": _safe_ratio(fill_m["retailer"], fill_d["retailer"]),
        "jitter": float(np.mean([np.mean(np.abs(np.diff(orders[a]))) if len(orders[a]) > 1 else 0.0
                                 for a in AGENTS])),
        "hold_total": hold_total, "back_total": back_total,
        "hold_frac": _safe_ratio(hold_total, hold_total + back_total),
        # per-stage dicts (keyed by agent name)
        "bw_stage": bw_stage, "ovar": ovar, "dvar": dvar,
        "bw_cum": bw_cum, "nsamp": nsamp, "mean_netstock": mean_netstock, "total_c": total_c,
        "mean_inv": mean_inv, "mean_back": mean_back,
        "serv_alpha": serv_alpha, "fill_beta_s": fill_beta_s,
        "stockout_freq": stockout_freq, "hold_c": hold_c, "back_c": back_c,
    }
    if trace:
        out["trace"] = {"dhat": np.array(tr_dhat), "msg": np.array(tr_msg), "cust": np.array(tr_cust)}
    return out


def cvar(costs, alpha):
    c = np.sort(np.asarray(costs))[::-1]
    k = max(1, int(np.ceil(alpha * len(c))))
    return float(c[:k].mean())


def evaluate(policy, env, episodes, trace=False):
    return [run_episode(policy, env, SEED_BASE + e, trace=trace) for e in range(episodes)]


# ==============================================================================
# Regime-uncertainty table (C1): per-lambda cost + Gap_Recovered vs BAR/CEILING
# ==============================================================================
def regime_uncertainty(ckpt, episodes, bar, ceiling, lambdas=HELDOUT_LAMBDAS):
    print(f"\n  regime-uncertainty (C1)   BAR={bar:.0f}  CEILING={ceiling:.0f}   "
          f"[{episodes} eps/lambda, CRN seeds {HELDOUT_SEED_BASE}+]")
    print(f"    {'lambda':>7}{'mean cost':>12}{'S_mean':>9}")
    per = {}
    for lam in lambdas:
        env = DemandRandomizedBeerGame({**ENV_BASE, "demand_type": "poisson"},
                                       lam_lo=float(lam), lam_hi=float(lam), p_shift=0.0)
        pol = DracoV4Policy(ckpt, env, ablate=False)
        costs, smeans = [], []
        for e in range(episodes):
            r = run_episode(pol, env, HELDOUT_SEED_BASE + e)
            costs.append(r["cost"]); smeans.append(np.mean(pol.last_S))
        per[lam] = float(np.mean(costs))
        print(f"    {lam:>7g}{per[lam]:>12.1f}{np.mean(smeans):>9.1f}")
    mean_cost = float(np.mean(list(per.values())))
    gap = (bar - mean_cost) / max(1e-6, bar - ceiling)
    verdict = "BEATS the fixed bar (C1)" if gap > 0 else "below the fixed bar (no C1 yet)"
    print(f"    {'MEAN':>7}{mean_cost:>12.1f}   Gap_Recovered={gap:+.3f}  ({verdict})")
    return {"per_lambda": per, "mean_cost": mean_cost, "gap_recovered": gap}


# ==============================================================================
# Message analysis (Study 2): does the channel carry demand, and does it help upstream?
# ==============================================================================
def message_analysis(ckpt, episodes=40, scenario="poisson"):
    env = BeerGameParallelEnv({**ENV_BASE, "demand_type": scenario})
    pol_on = DracoV4Policy(ckpt, env, ablate=False)
    if not pol_on.use_comm:
        print("\n  message analysis: checkpoint has use_comm=false. Skipped.")
        return None
        
    if pol_on.msg_heads is None:
        print("\n  message analysis: checkpoint has no comm channel (use_comm was false). skipped.")
        return None
    pol_off = DracoV4Policy(ckpt, env, ablate=True)

    msg_all, cust_all = [], []
    dhat_on, dhat_off, cust_on = [], [], []
    for e in range(episodes):
        r_on = run_episode(pol_on, env, SEED_BASE + e, trace=True)
        r_off = run_episode(pol_off, env, SEED_BASE + e, trace=True)
        tr = r_on["trace"]
        msg_all.append(tr["msg"]); cust_all.append(tr["cust"])
        dhat_on.append(tr["dhat"]); cust_on.append(tr["cust"])
        dhat_off.append(r_off["trace"]["dhat"])

    msg = np.concatenate(msg_all, axis=0)                 # [T_tot, N, msg_dim]
    cust = np.concatenate(cust_all, axis=0)               # [T_tot]
    print(f"\n  message analysis ({scenario}, topology={pol_on.comm_topology}, {episodes} eps)")

    # (1) per-channel saturation + correlation with the SENDER's demand proxy (customer demand,
    #     the shared regime signal). High |corr| => the channel encodes the demand/regime.
    print(f"    {'sender':<13}{'|msg| mean':>11}{'sat>0.9':>9}{'corr(ch0,demand)':>18}")
    for i, a in enumerate(AGENTS):
        ch = msg[:, i, :]
        sat = float(np.mean(np.abs(ch) > 0.9))
        absm = float(np.mean(np.abs(ch)))
        c0 = ch[:, 0]
        corr = float(np.corrcoef(c0, cust)[0, 1]) if np.std(c0) > 1e-8 else float("nan")
        print(f"    {a:<13}{absm:>11.3f}{sat:>9.2f}{corr:>18.3f}")

    # (2) see-through-bullwhip: upstream demand-tracking error vs the true customer demand,
    #     WITH vs WITHOUT messages. delta = err_off - err_on; >0 => messages help upstream.
    if pol_on.has_dhat:
        don = np.concatenate(dhat_on, axis=0)             # [T_tot, N]
        doff = np.concatenate(dhat_off, axis=0)
        cst = np.concatenate(cust_on, axis=0)
        print(f"    {'upstream':<13}{'err_msgON':>11}{'err_msgOFF':>12}{'delta(help)':>13}")
        for i, a in enumerate(AGENTS):
            if a == "retailer":
                continue                                  # retailer observes demand directly
            e_on = float(np.nanmean(np.abs(don[:, i] - cst)))
            e_off = float(np.nanmean(np.abs(doff[:, i] - cst)))
            print(f"    {a:<13}{e_on:>11.2f}{e_off:>12.2f}{e_off - e_on:>+13.2f}")
        print("    (delta>0 = the shared demand signal lowers upstream forecast error: "
              "the Lee-et-al see-through-bullwhip channel is active.)")
    else:
        print("    (mlp head has no d_hat readout; skip the upstream-forecast diagnostic.)")
    return {"msg": msg, "cust": cust}


# ==============================================================================
# Belief diagnostics (the CORRECT way to test a dhat channel) + OM dashboards
# ==============================================================================
def _mutual_info(dhat_samples, lambda_labels, bins=8):
    """MI(d_hat ; lambda) in bits, and MI/H(lambda). Robust positive-signalling metric
    (Lowe et al. 2019): computed against the REGIME label, immune to the stationary
    within-episode noise that makes raw corr(msg, per-step demand) read ~0."""
    dh = np.asarray(dhat_samples, float); lab = np.asarray(lambda_labels, float)
    ok = np.isfinite(dh)
    dh, lab = dh[ok], lab[ok]
    if dh.size < 10 or np.std(dh) < 1e-9:
        return 0.0, 0.0
    edges = np.quantile(dh, np.linspace(0, 1, bins + 1)); edges[-1] += 1e-6
    dbin = np.clip(np.digitize(dh, edges[1:-1]), 0, bins - 1)
    labs = np.unique(lab)
    joint = np.zeros((bins, len(labs)))
    for k, lv in enumerate(labs):
        m = lab == lv
        for b in range(bins):
            joint[b, k] = np.sum(dbin[m] == b)
    s = joint.sum()
    if s < 1:
        return 0.0, 0.0
    joint /= s
    pd = joint.sum(1, keepdims=True); pl = joint.sum(0, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        mi = float(np.nansum(joint * np.log2((joint + 1e-12) / (pd * pl + 1e-12))))
    hl = float(-np.nansum(pl * np.log2(pl + 1e-12)))
    return max(mi, 0.0), (max(mi, 0.0) / hl if hl > 1e-9 else 0.0)


def belief_calibration(ckpt, episodes=20, lambdas=HELDOUT_LAMBDAS):
    """POSITIVE SIGNALLING. Regress per-episode mean d_hat on the true lambda ACROSS regimes
    (the cross-regime test, not the broken within-episode corr). slope~1/intercept~0/R^2~1 and
    high MI => the broadcast faithfully encodes the demand regime."""
    print("\n  belief calibration  (cross-regime: per-episode mean d_hat vs true lambda)")
    per_ech = {a: {"lam": [], "dh": []} for a in AGENTS}
    pool = {a: {"dh": [], "lam": []} for a in AGENTS}
    probe = DracoV4Policy(ckpt, BeerGameParallelEnv({**ENV_BASE, "demand_type": "poisson"}))
    if not probe.has_dhat:
        print("    (mlp head has no d_hat readout; skipped.)"); return None
    for lam in lambdas:
        env = DemandRandomizedBeerGame({**ENV_BASE, "demand_type": "poisson"},
                                       lam_lo=float(lam), lam_hi=float(lam), p_shift=0.0)
        pol = DracoV4Policy(ckpt, env, ablate=False)
        for e in range(episodes):
            r = run_episode(pol, env, HELDOUT_SEED_BASE + e, trace=True)
            dh = r["trace"]["dhat"]                                  # [T, N]
            for i, a in enumerate(AGENTS):
                col = dh[:, i]; col = col[np.isfinite(col)]
                if col.size:
                    per_ech[a]["lam"].append(float(lam)); per_ech[a]["dh"].append(float(col.mean()))
                    pool[a]["dh"].extend(col.tolist()); pool[a]["lam"].extend([float(lam)] * col.size)
    print(f"    {'echelon':<13}{'slope':>8}{'intercept':>11}{'R^2':>8}{'mean_bias':>11}{'MI(bits)':>10}{'MI/H':>7}")
    res = {}
    for a in AGENTS:
        lam = np.asarray(per_ech[a]["lam"]); dh = np.asarray(per_ech[a]["dh"])
        if lam.size < 2 or np.std(lam) < 1e-9:
            continue
        slope, intercept = np.polyfit(lam, dh, 1)
        pred = slope * lam + intercept
        ss = float(np.sum((dh - pred) ** 2)); tot = float(np.sum((dh - dh.mean()) ** 2))
        r2 = 1.0 - ss / tot if tot > 1e-9 else float("nan")
        bias = float(np.mean(dh - lam))
        mi, mih = _mutual_info(pool[a]["dh"], pool[a]["lam"])
        res[a] = dict(slope=float(slope), intercept=float(intercept), r2=r2, bias=bias, mi=mi, mi_norm=mih)
        print(f"    {a:<13}{slope:>8.3f}{intercept:>11.2f}{r2:>8.3f}{bias:>+11.2f}{mi:>10.3f}{mih:>7.2f}")
    print("    (slope~1, R^2~1, MI high => d_hat encodes the regime -- proves the channel is ALIVE.")
    print("     persistent +bias => broadcast over-states demand, a concrete reason comm hurts at low lambda.)")
    return res


def adaptation_speed(ckpt, episodes=20, lambdas=HELDOUT_LAMBDAS, eps_abs=2.0):
    """META-RL ADAPTATION (RL^2 / VariBAD). Steps until |mean d_hat - lambda| < eps, comm ON vs OFF.
    The broadcast should converge UPSTREAM beliefs faster (clean signal injected early)."""
    print(f"\n  belief adaptation speed  (steps to |mean d_hat - lambda| < {eps_abs:.1f}; ON vs OFF)")
    probe = DracoV4Policy(ckpt, BeerGameParallelEnv({**ENV_BASE, "demand_type": "poisson"}))
    if not probe.has_dhat:
        print("    (mlp head has no d_hat; skipped.)"); return None

    def time_to_converge(ablate):
        per_ech = {a: [] for a in AGENTS}
        for lam in lambdas:
            env = DemandRandomizedBeerGame({**ENV_BASE, "demand_type": "poisson"},
                                           lam_lo=float(lam), lam_hi=float(lam), p_shift=0.0)
            pol = DracoV4Policy(ckpt, env, ablate=ablate)
            traj = [run_episode(pol, env, HELDOUT_SEED_BASE + e, trace=True)["trace"]["dhat"]
                    for e in range(episodes)]
            T = min(len(t) for t in traj)
            mean_dh = np.nanmean(np.stack([t[:T] for t in traj]), axis=0)       # [T, N]
            for i, a in enumerate(AGENTS):
                hit = np.where(np.abs(mean_dh[:, i] - lam) < eps_abs)[0]
                per_ech[a].append(float(hit[0]) if hit.size else float(T))
        return {a: float(np.mean(v)) for a, v in per_ech.items()}

    on, off = time_to_converge(False), time_to_converge(True)
    print(f"    {'echelon':<13}{'steps_ON':>10}{'steps_OFF':>11}{'speedup':>10}")
    for a in AGENTS:
        print(f"    {a:<13}{on[a]:>10.1f}{off[a]:>11.1f}{off[a] - on[a]:>+10.1f}")
    print("    (speedup>0 = comm converges the belief faster; expected for UPSTREAM echelons.)")
    return {"on": on, "off": off}


def message_responsiveness(ckpt, episodes=10, lambdas=HELDOUT_LAMBDAS, delta=0.05):
    """CAUSAL POSITIVE LISTENING (Jaques et al. 2019). dS / d(broadcast demand) by counterfactually
    perturbing the incoming message and re-querying the actor. >0 => the receiver raises its
    order-up-to level when told demand is higher -- it acts on message CONTENT, not just presence."""
    print("\n  message responsiveness  (dS / d(broadcast demand), counterfactual probe)")
    probe = DracoV4Policy(ckpt, BeerGameParallelEnv({**ENV_BASE, "demand_type": "poisson"}))
    if not probe.use_comm:
        print("    (use_comm=false; no channel to probe.)"); return None
    if not probe.has_dhat:
        print("    (structured head required for the dhat responsiveness probe; skipped.)"); return None
    elas = {a: [] for a in AGENTS}
    for lam in lambdas:
        env = DemandRandomizedBeerGame({**ENV_BASE, "demand_type": "poisson"},
                                       lam_lo=float(lam), lam_hi=float(lam), p_shift=0.0)
        pol = DracoV4Policy(ckpt, env, ablate=False)
        for e in range(episodes):
            obs, _ = env.reset(seed=HELDOUT_SEED_BASE + e); pol.reset()
            while True:
                acts = pol.act(obs)
                _o, _z, inc = pol.last_ctx
                for i, a in enumerate(AGENTS):
                    base = float(inc[i].reshape(-1)[0].item())
                    s_hi = pol.probe_S(i, base + delta); s_lo = pol.probe_S(i, base - delta)
                    elas[a].append((s_hi - s_lo) / (2.0 * delta * 100.0))      # per UNIT of broadcast demand
                obs, _r, _t, truncs, _inf = env.step({a: [acts[a]] for a in AGENTS})
                if any(truncs.values()):
                    break
    print(f"    {'echelon':<13}{'dS/d(demand)':>14}")
    for a in AGENTS:
        print(f"    {a:<13}{float(np.mean(elas[a])):>14.3f}")
    print("    (>0 = raises order-up-to S when told demand is higher: it listens. ~0 at the retailer")
    print("     is expected -- it ignores the channel and observes demand directly.)")
    return {a: float(np.mean(v)) for a, v in elas.items()}


def per_stage_dashboard(ckpt, episodes, scenario, eps=None):
    """The full per-echelon OM dashboard (Chen et al. 2000; Lee et al. 1997; Disney-Towill 2003):
    local + cumulative bullwhip, net-stock amplification, on-hand/backlog/net-stock, Type-1 service,
    Type-2 fill, stockout frequency, and per-echelon total cost. Pass `eps` to reuse a rollout."""
    if eps is None:
        env = BeerGameParallelEnv({**ENV_BASE, "demand_type": scenario})
        eps = evaluate(DracoV4Policy(ckpt, env, ablate=False), env, episodes)
    n = len(eps)

    def m(key, a): return float(np.nanmean([e[key][a] for e in eps]))
    print(f"\n  per-stage dashboard ({scenario}, {n} eps)")
    print(f"    {'echelon':<13}{'BW_loc':>8}{'BW_cum':>8}{'NSAmp':>8}{'inv':>7}{'back':>7}{'netstk':>8}"
          f"{'alpha':>7}{'beta':>7}{'P(stk)':>8}{'total$':>9}")
    for a in AGENTS:
        print(f"    {a:<13}{m('bw_stage',a):>8.2f}{m('bw_cum',a):>8.2f}{m('nsamp',a):>8.2f}"
              f"{m('mean_inv',a):>7.1f}{m('mean_back',a):>7.1f}{m('mean_netstock',a):>8.1f}"
              f"{m('serv_alpha',a):>7.2f}{m('fill_beta_s',a):>7.2f}{m('stockout_freq',a):>8.2f}{m('total_c',a):>9.0f}")
    cpd = float(np.nanmean([e["cost_per_demand"] for e in eps]))
    hf = float(np.nanmean([e["hold_frac"] for e in eps]))
    bws = float(np.nanmean([e["bw_slope"] for e in eps]))
    print(f"    chain: cost/unit-demand={cpd:.2f}  holding-cost-share={hf:.2f}  "
          f"variance-amplification-slope={bws:+.2f}")
    print("    (BW_loc=Var(ord)/Var(own-incoming); BW_cum=Var(ord)/Var(customer demand) [Lee/Chen];")
    print("     NSAmp=Var(net stock)/Var(customer demand) [Disney-Towill]; total$=holding+backorder at that stage.)")


def comm_value_decomposition(ckpt, episodes=40, scenario="poisson"):
    """Decompose the comm effect (ON vs OFF, CRN-paired): per-stage bullwhip, on-hand/backlog,
    holding/backorder cost split, and upstream forecast error vs CUSTOMER demand (the Lee target)."""
    env = BeerGameParallelEnv({**ENV_BASE, "demand_type": scenario})
    pol_on = DracoV4Policy(ckpt, env, ablate=False)
    if not pol_on.use_comm:
        print(f"\n  comm decomposition ({scenario}): use_comm=false. skipped."); return None
    pol_off = DracoV4Policy(ckpt, env, ablate=True)
    on = [run_episode(pol_on, env, SEED_BASE + e, trace=True) for e in range(episodes)]
    off = [run_episode(pol_off, env, SEED_BASE + e, trace=True) for e in range(episodes)]

    def agg(rs, key, a): return float(np.nanmean([r[key][a] for r in rs]))
    print(f"\n  comm decomposition ({scenario}, {episodes} eps, ON vs OFF paired)")
    print(f"    {'echelon':<13}{'BW_ON':>8}{'BW_OFF':>8}{'inv_ON':>8}{'inv_OFF':>8}{'back_ON':>8}{'back_OFF':>9}")
    for a in AGENTS:
        print(f"    {a:<13}{agg(on,'bw_stage',a):>8.2f}{agg(off,'bw_stage',a):>8.2f}"
              f"{agg(on,'mean_inv',a):>8.1f}{agg(off,'mean_inv',a):>8.1f}"
              f"{agg(on,'mean_back',a):>8.1f}{agg(off,'mean_back',a):>9.1f}")
    h_on = np.mean([r["hold_total"] for r in on]); h_off = np.mean([r["hold_total"] for r in off])
    b_on = np.mean([r["back_total"] for r in on]); b_off = np.mean([r["back_total"] for r in off])
    print(f"    cost split (ON->OFF): holding {h_on:.0f}->{h_off:.0f}   backorder {b_on:.0f}->{b_off:.0f}")
    if pol_on.has_dhat:
        don = np.concatenate([r["trace"]["dhat"] for r in on], 0)
        doff = np.concatenate([r["trace"]["dhat"] for r in off], 0)
        cst = np.concatenate([r["trace"]["cust"] for r in on], 0)
        print(f"    {'upstream':<13}{'fErr_ON':>9}{'fErr_OFF':>10}{'delta(help)':>13}  (target = CUSTOMER demand)")
        for i, a in enumerate(AGENTS):
            if a == "retailer":
                continue
            e_on = float(np.nanmean(np.abs(don[:, i] - cst)))
            e_off = float(np.nanmean(np.abs(doff[:, i] - cst)))
            print(f"    {a:<13}{e_on:>9.2f}{e_off:>10.2f}{e_off - e_on:>+13.2f}")
        print("    (delta>0 = sharing the clean retailer demand cuts upstream forecast error = Lee see-through-bullwhip.")
        print("     delta<0 in a STABLE regime is the real Axsater-Rosling result, not a bug: local info already suffices.)")
    return None


# ==============================================================================
# main
# ==============================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--cvar", type=float, default=0.2, help="tail level for CVaR cost")
    ap.add_argument("--bar", type=float, default=4726.0, help="fixed-policy BAR (baselines.py regime)")
    ap.add_argument("--ceiling", type=float, default=2202.0, help="per-lambda CEILING (baselines.py regime)")
    ap.add_argument("--regime-episodes", type=int, default=20, help="episodes per held-out lambda")
    ap.add_argument("--messages", action="store_true", help="run the Study-2 message analysis")
    ap.add_argument("--full", action="store_true",
                    help="generate EVERYTHING: per-stage OM dashboard, belief calibration+MI, "
                         "adaptation speed, message responsiveness, comm decomposition (all scenarios)")
    ap.add_argument("--diag-episodes", type=int, default=20, help="episodes for the belief diagnostics")
    args = ap.parse_args()

    ckpt = torch.load(args.ckpt, map_location=DEVICE, weights_only=False)
    cfg = ckpt.get("config", {}).get("agent", {})
    print(f"\nDRACO v4 eval  |  ckpt={os.path.basename(args.ckpt)}  |  head={cfg.get('actor_head')}  "
          f"|  encoder={cfg.get('encoder_type','gru')}  |  comm={cfg.get('use_comm')}"
          f"/{cfg.get('comm_topology','neighbor')}  |  episodes={args.episodes}\n")

    # ---- standard benchmark across the fixed regimes ----
    header = (f"{'scenario':<15}{'mean cost':>12}{'CVaR cost':>12}{'bullwhip':>11}"
              f"{'srv-alpha':>11}{'fill-beta':>11}{'jitter':>9}   comm value")
    print(header); print("-" * len(header))
    cost_by_scenario = {}
    eps_by_scenario = {}
    for scenario in SCENARIOS:
        env = BeerGameParallelEnv({**ENV_BASE, "demand_type": scenario})
        eps = evaluate(DracoV4Policy(ckpt, env, ablate=False), env, args.episodes)
        costs = np.array([e["cost"] for e in eps])
        cost_by_scenario[scenario] = costs
        eps_by_scenario[scenario] = eps
        abl = evaluate(DracoV4Policy(ckpt, env, ablate=True), env, args.episodes)
        c_abl = np.array([e["cost"] for e in abl])
        with np.errstate(invalid="ignore", divide="ignore"):
            comm_value = float(np.nanmean((c_abl - costs) / np.where(costs == 0, np.nan, costs)) * 100.0)
        try:
            p = float(stats.wilcoxon(c_abl, costs).pvalue) if not np.all(c_abl - costs == 0) else 1.0
        except Exception:
            p = float("nan")
        print(f"{scenario:<15}{costs.mean():>12.1f}{cvar(costs, args.cvar):>12.1f}"
              f"{np.nanmean([e['bw_overall'] for e in eps]):>11.2f}"
              f"{np.mean([e['ret_service_alpha'] for e in eps]):>11.2f}"
              f"{np.nanmean([e['fill_beta'] for e in eps]):>11.2f}"
              f"{np.mean([e['jitter'] for e in eps]):>9.1f}"
              f"   {comm_value:+.1f}%  (p={p:.1e})")
    print("\ncomm value = % cost change when messages are zeroed (paired, same seeds);"
          " positive => communication helps.")

    # ---- cost distribution & risk (mean +/- std, CV, percentiles, worst-case) ----
    #   point estimates hide risk; top-tier reporting gives dispersion + tails, not just the mean.
    print(f"\n  cost distribution & risk ({args.episodes} eps)")
    print(f"    {'scenario':<15}{'mean':>10}{'std':>9}{'CV':>7}{'median':>10}{'p90':>10}"
          f"{'p95':>10}{'max':>10}{f'CVaR{int(args.cvar*100)}':>10}")
    for scenario in SCENARIOS:
        c = cost_by_scenario[scenario]
        cv = float(c.std() / c.mean()) if c.mean() else float("nan")
        print(f"    {scenario:<15}{c.mean():>10.1f}{c.std():>9.1f}{cv:>7.2f}"
              f"{np.median(c):>10.1f}{np.percentile(c,90):>10.1f}{np.percentile(c,95):>10.1f}"
              f"{c.max():>10.1f}{cvar(c, args.cvar):>10.1f}")
    print("    (CV=std/mean = relative risk; a low mean with high CV/max is a fragile policy. Report mean +/- std.)")

    # ---- per-stage dashboard (DEFAULT: reuses the rollouts above, so it's free) ----
    for scenario in SCENARIOS:
        per_stage_dashboard(ckpt, args.episodes, scenario, eps=eps_by_scenario[scenario])

    # ---- regime-uncertainty (C1) ----
    regime_uncertainty(ckpt, args.regime_episodes, args.bar, args.ceiling)

    # ---- message analysis (Study 2) ----
    if args.messages or args.full:
        message_analysis(ckpt, episodes=min(40, args.episodes))

    # ---- FULL diagnostic suite: belief signalling/listening + comm decomposition ----
    # each wrapped so one failure (e.g. an mlp/no-comm ckpt) can't swallow the rest.
    if args.full:
        print("\n" + "=" * 94)
        print("FULL DIAGNOSTICS  (belief signalling/listening + comm decomposition)")
        print("=" * 94)
        for name, fn in [
            ("belief_calibration", lambda: belief_calibration(ckpt, episodes=args.diag_episodes)),
            ("adaptation_speed",   lambda: adaptation_speed(ckpt, episodes=args.diag_episodes)),
            ("message_responsiveness", lambda: message_responsiveness(ckpt, episodes=max(5, args.diag_episodes // 2))),
        ]:
            try:
                fn()
            except Exception as e:
                print(f"\n  [{name}] skipped (error: {type(e).__name__}: {e})")
        for scenario in SCENARIOS:
            try:
                comm_value_decomposition(ckpt, episodes=min(40, args.episodes), scenario=scenario)
            except Exception as e:
                print(f"\n  [comm_value_decomposition/{scenario}] skipped (error: {type(e).__name__}: {e})")
    print()


if __name__ == "__main__":
    main()