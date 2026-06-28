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
import json
import argparse
import numpy as np
import torch
from scipy import stats

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from envs.beer_game_env import BeerGameParallelEnv
from agents.draco_v4 import (
    make_encoder, make_actor, BaseStockActor, MessageHead, DemandRandomizedBeerGame, plugin_rate,
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

# ==============================================================================
# READING GUIDE -- what this script computes and which question each part answers.
#
# It loads a trained checkpoint into DracoV4Policy (topology rebuilt from the ckpt so messages
# route exactly as trained), rolls deterministic episodes, and prints:
#
#   * STANDARD BENCHMARK (per regime) -- mean/CVaR cost, bullwhip, service, fill, jitter, and the
#     zero-message "comm value" (cost change when messages are ablated, paired Wilcoxon).
#   * REGIME-UNCERTAINTY (C1)  -> regime_uncertainty(): per-lambda cost + Gap_Recovered vs the
#     BAR/Oracle/Bayes references. THE headline table. (Study 1/2)
#   * HELD-OUT-FAMILY eval     -> heldout_family_eval(): AR(1)/NegBin distributional shift vs the
#     family-appropriate optimum (AR1-opt / Bayes). (M2)  [--families]
#   * MESSAGE / BELIEF DIAGNOSTICS (Study 3, comm checkpoints only): does the channel carry the
#     demand signal and does it help upstream? -> belief_calibration (positive signalling: is d_hat
#     a faithful regime readout?), adaptation_speed (does comm converge beliefs faster?),
#     message_responsiveness (positive listening: dS/d(message) -- does the receiver ACT on content?),
#     comm_value_decomposition (per-stage bullwhip/cost split + upstream forecast-error delta).
#   * PRODUCER mode -> dump_c1(): writes per-seed {lambda: cost} for scripts/c1_stats.py.  [--dump-c1]
#
# TWO SEED SPACES (do not conflate):
#   SEED_BASE=2000          -> the standard-benchmark/message rollouts (their own held-out block).
#   HELDOUT_SEED_BASE=1e5   -> the C1 / family rollouts; == baselines.py SEED_BASE, so DRACO and the
#                              reference rungs are scored on the SAME demand draws (common random numbers).
#
# OM METRIC GLOSSARY (per-stage dashboard):
#   BW_loc  = Var(orders)/Var(own incoming)            local order-variance amplification
#   BW_cum  = Var(orders)/Var(CUSTOMER demand)         cumulative bullwhip vs the true source (Lee/Chen 2000)
#   NSAmp   = Var(net stock)/Var(customer demand)      net-stock amplification (Disney-Towill 2003)
#   alpha   = P(no backlog this step)                  Type-1 / cycle service level
#   beta    = demand_met / demand                      Type-2 / fill rate
# ==============================================================================


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
        # belief_mode (ablation #8): if the checkpoint used a plug-in belief, eval must use it too
        # (the encoder is untrained in that case), so d_hat matches training.
        self.belief_mode = str(cfg.get("belief_mode", "encoder")).lower()
        self.plugin_prior_mean = float(cfg.get("plugin_prior_mean", cfg.get("demand_init", 8.0)))
        self.plugin_prior_strength = float(cfg.get("plugin_prior_strength", 1.0))
        self.plugin_eta = float(cfg.get("plugin_eta", 0.2))

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
        d_hat_override = None
        if self.belief_mode == "encoder":
            mu, _, _ = self.encoder.forward_sequence(obs_seq, msg_seq)
            z_t = mu[-1]
            z_act = z_t if self.use_context else torch.zeros_like(z_t)
        else:                                            # plug-in belief (ablation #8): match training
            z_t = torch.zeros(self.N, self.z_dim, device=DEVICE)
            z_act = z_t
            d_hat_override = plugin_rate(obs_seq[..., 3], self.belief_mode, self.plugin_prior_mean,
                                         self.plugin_prior_strength, self.plugin_eta)[-1]   # [N,1]

        S = torch.zeros(self.N, 1, device=DEVICE)
        dhat = np.full(self.N, np.nan)
        for i in range(self.N):
            dho = d_hat_override[i:i + 1] if d_hat_override is not None else None
            s_mu, s_std = self.actors[i](o_t[i:i + 1], z_act[i:i + 1], incoming[i:i + 1], d_hat_override=dho)
            S[i] = s_mu if self.deterministic else torch.distributions.Normal(s_mu, s_std).sample()
            if self.has_dhat:
                dhat[i] = (float(d_hat_override[i].item()) if d_hat_override is not None
                           else float(self.actors[i].demand_estimate(z_act[i:i + 1]).reshape(-1)[0].item()))

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
def regime_uncertainty(ckpt, episodes, bar, ceiling, lambdas=HELDOUT_LAMBDAS, bayes=None):
    print(f"\n  regime-uncertainty (C1)   BAR={bar:.0f}  CEILING={ceiling:.0f}"
          + (f"  BAYES={bayes:.0f}" if bayes is not None else "")
          + f"   [{episodes} eps/lambda, CRN seeds {HELDOUT_SEED_BASE}+]")
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
    out = {"per_lambda": per, "mean_cost": mean_cost, "gap_recovered": gap, "bayes": bayes}
    if bayes is not None:
        # the HEADLINE comparison: DRACO vs the analytic Bayes-adaptive policy (Scarf/Azoury),
        # not the static base-stock. Bayes_Gap = fraction of the Bayes->oracle headroom recovered.
        # SECONDARY check ("beats naive adaptation"): Bayes is the single-stage-optimal adaptive
        # policy, which BULLWHIPS in this multi-echelon cost -> a naive-adaptation FLOOR, not the
        # headline bar. The headline is Gap_Recovered vs the static BAR + Oracle (above).
        bgap = (bayes - mean_cost) / max(1e-6, bayes - ceiling)
        bverdict = "beats the naive-adaptation floor (Bayes)" if mean_cost < bayes else "below the Bayes floor"
        print(f"    {'vs BAYES':>7}{bayes:>12.1f}   Bayes_Gap={bgap:+.3f}  ({bverdict})")
        out["bayes_gap_recovered"] = bgap
    return out


def dump_c1(ckpt, episodes, out_dir, lambdas=HELDOUT_LAMBDAS, seed=None):
    """Producer for c1_stats: write {lambda: mean_cost} for THIS checkpoint to out_dir/seed{S}.json.
    Scored on the EVAL seeds (HELDOUT_SEED_BASE+e) == the seeds baselines.py reports the rungs on,
    so DRACO and the references are CRN-comparable. S defaults to the checkpoint's training seed.
    Run this once per Phase-2 checkpoint, then `python scripts/c1_stats.py report`."""
    if seed is None:
        seed = ckpt.get("config", {}).get("seed", 0)
    per, per_bw = {}, {}
    for lam in lambdas:
        env = DemandRandomizedBeerGame({**ENV_BASE, "demand_type": "poisson"},
                                       lam_lo=float(lam), lam_hi=float(lam), p_shift=0.0)
        pol = DracoV4Policy(ckpt, env, ablate=False)
        rs = [run_episode(pol, env, HELDOUT_SEED_BASE + e) for e in range(episodes)]
        per[float(lam)] = float(np.mean([r["cost"] for r in rs]))
        # per-echelon cumulative bullwhip BW_cum = Var(orders)/Var(customer demand), avg over episodes
        per_bw[float(lam)] = {a: float(np.nanmean([r["bw_cum"][a] for r in rs])) for a in AGENTS}
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"seed{int(seed)}.json")
    with open(path, "w") as f:
        json.dump(per, f, indent=2)
    bw_path = os.path.join(out_dir, f"seed{int(seed)}_bw.json")          # consumed by run_confirmatory_report
    with open(bw_path, "w") as f:
        json.dump(per_bw, f, indent=2)
    print(f"\n  [dump-c1] wrote {path} (+ {os.path.basename(bw_path)})")
    print(f"    lambdas {[f'{l:g}' for l in lambdas]}  mean_cost "
          f"{[round(per[float(l)],1) for l in lambdas]}  ({episodes} eps/lambda)")
    return path


def _roll_orders_baseline(policy, env, seed):
    """Roll a baselines.py policy (act(obs, env) -> {agent: [frac]}) and return per-echelon cumulative
    bullwhip BW_cum = Var(orders)/Var(customer demand) plus total cost. Mirrors the BW_cum definition
    used for DRACO in run_episode()."""
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


def bullwhip_comparison(ckpt, episodes=40, lambdas=HELDOUT_LAMBDAS, refs_json=None, prior_mean=14.0):
    """FIRST-CLASS bullwhip result (review #13 / the S2 story): order-variance amplification
    BW_cum = Var(orders)/Var(customer demand) per echelon, for DRACO vs the static base-stock vs the
    Bayes naive-adaptation floor, on the held-out regimes. A static base-stock is ~bullwhip-free
    (BW~1); adaptive policies amplify upstream. The question DRACO must answer: can it ADAPT the
    regime while keeping amplification low (i.e. beat static cost without static's bullwhip)?
    Static-BAR levels are read from the refs JSON `meta.bar_levels` (single source of truth)."""
    try:
        from scripts.baselines import BaseStockPolicy, make_bayes_rung
    except Exception as e:
        print(f"\n  bullwhip comparison skipped (import: {type(e).__name__}: {e})")
        return None
    bar_levels = None
    if refs_json and os.path.exists(refs_json):
        try:
            bar_levels = json.load(open(refs_json)).get("meta", {}).get("bar_levels")
        except Exception:
            bar_levels = None

    def _agg_baseline(make_pol):
        bws = {a: [] for a in AGENTS}
        costs = []
        for lam in lambdas:
            env = DemandRandomizedBeerGame({**ENV_BASE, "demand_type": "poisson"},
                                           lam_lo=float(lam), lam_hi=float(lam), p_shift=0.0)
            for e in range(episodes):
                bw, c = _roll_orders_baseline(make_pol(), env, HELDOUT_SEED_BASE + e)
                for a in AGENTS:
                    bws[a].append(bw[a])
                costs.append(c)
        return {a: float(np.nanmean(bws[a])) for a in AGENTS}, float(np.mean(costs))

    def _agg_draco():
        bws = {a: [] for a in AGENTS}
        costs = []
        for lam in lambdas:
            env = DemandRandomizedBeerGame({**ENV_BASE, "demand_type": "poisson"},
                                           lam_lo=float(lam), lam_hi=float(lam), p_shift=0.0)
            pol = DracoV4Policy(ckpt, env, ablate=False)
            for e in range(episodes):
                r = run_episode(pol, env, HELDOUT_SEED_BASE + e)
                for a in AGENTS:
                    bws[a].append(r["bw_cum"][a])
                costs.append(r["cost"])
        return {a: float(np.nanmean(bws[a])) for a in AGENTS}, float(np.mean(costs))

    print(f"\n  bullwhip comparison  (BW_cum = Var(orders)/Var(customer demand); review #13, {episodes} eps/lambda)")
    print(f"    {'policy':<16}" + "".join(f"{('BW_' + a[:4]):>9}" for a in AGENTS) + f"{'mean cost':>11}")
    rows = {}
    if bar_levels is not None:
        bw, c = _agg_baseline(lambda: BaseStockPolicy(bar_levels))
        rows["static-BAR"] = (bw, c)
    else:
        print("    (static-BAR row skipped: no meta.bar_levels in the refs JSON; run baselines.py regime)")
    rows["Bayes-floor"] = _agg_baseline(lambda: make_bayes_rung(h=H_COST, b=B_COST, prior_mean=prior_mean))
    rows["DRACO"] = _agg_draco()
    for name, (bw, c) in rows.items():
        print(f"    {name:<16}" + "".join(f"{bw[a]:>9.2f}" for a in AGENTS) + f"{c:>11.1f}")
    print("    (static base-stock is ~bullwhip-free [BW~1]; the Bayes floor amplifies upstream; the")
    print("     S2 claim is that DRACO adapts the regime while keeping BW_cum low -> 'adapt without bullwhip'.)")
    return rows


def action_diagnostics(ckpt, episodes=20, lambdas=(8.0, 22.0)):
    """Saturation/clipping diagnostics for the continuous-S -> order = clip(S - IP, 0, max_order)
    -> round map (review #11). There is NO policy-gradient bias (the Gaussian action IS S; the clip
    and round are deterministic ENV transforms downstream), but heavy saturation flattens gradients
    and is worth monitoring. Reports, per echelon: fraction of steps clipped at 0 (desired S <= IP),
    fraction at the order cap, and the desired-order (S - IP) distribution. Low+high lambda surface
    over-stock (clip@0) vs cap-binding (at-cap)."""
    mo = float(ENV_BASE["max_order"])
    print(f"\n  action saturation diagnostics  ({episodes} eps/lambda, lambdas {[f'{l:g}' for l in lambdas]})")
    print(f"    {'echelon':<13}{'clip@0 %':>10}{'at-cap %':>10}{'desire p10':>11}{'desire med':>11}{'desire p90':>11}")
    agg = {a: [] for a in AGENTS}
    for lam in lambdas:
        env = DemandRandomizedBeerGame({**ENV_BASE, "demand_type": "poisson"},
                                       lam_lo=float(lam), lam_hi=float(lam), p_shift=0.0)
        pol = DracoV4Policy(ckpt, env, ablate=False)
        for e in range(episodes):
            obs, _ = env.reset(seed=HELDOUT_SEED_BASE + e)
            pol.reset()
            while True:
                acts = pol.act(obs)                                 # sets pol.last_S
                for i, a in enumerate(AGENTS):
                    o = obs[a]; ip = float(o[0]) - float(o[1]) + float(o[2])
                    agg[a].append(float(pol.last_S[i]) - ip)        # desired order S - IP (pre-clip)
                obs, _, _, trunc, _ = env.step({a: [acts[a]] for a in AGENTS})
                if any(trunc.values()):
                    break
    out = {}
    for a in AGENTS:
        d = np.asarray(agg[a], float)
        clip0, cap = float(np.mean(d <= 0.0)), float(np.mean(d >= mo))
        out[a] = {"clip0_frac": clip0, "cap_frac": cap, "p10": float(np.percentile(d, 10)),
                  "median": float(np.median(d)), "p90": float(np.percentile(d, 90))}
        print(f"    {a:<13}{100*clip0:>10.1f}{100*cap:>10.1f}{out[a]['p10']:>11.1f}"
              f"{out[a]['median']:>11.1f}{out[a]['p90']:>11.1f}")
    print("    (high clip@0 = often over-stocked/idle; high at-cap = demand exceeds the order cap -> "
          "raise max_order or check saturation. No PG bias -- S is the sampled action.)")
    return out


def heldout_family_eval(ckpt, episodes=50, mu=12.0, ar1_rhos=(0.0, 0.3, 0.6, 0.9), nb_disps=(2.0, 4.0, 8.0)):
    """M2.4 -- held-out-FAMILY eval (the distributional-shift test the 'robustness' name promises).
    Scores DRACO on AR(1)-rho and NegBin-dispersion demand against the FAMILY-APPROPRIATE optimum:
      * AR(1) rows -> the MMFE/AR(1)-optimal base-stock (AR1BaseStockPolicy) with the env's rho;
      * NegBin rows -> the Gamma-Poisson Bayes policy (NegBin IS the Gamma-Poisson predictive, so
        Bayes is its correct model).
    Using the right optimum per family is essential: the Gamma-Poisson Bayes rung is NOT optimal under
    autocorrelation, so 'beats Bayes' on AR(1) would be a category error. CRN seeds match baselines.py."""
    try:
        from scripts.demand_families import make_demand_family_envs, make_heldout_family_envs
        from scripts.baselines import (AdaptiveForecastPolicy, make_ar1_rung, make_bayes_rung,
                                        rollout as bl_rollout)
    except Exception as e:
        print(f"\n  held-out-family eval skipped (import: {type(e).__name__}: {e})")
        return None
    AR1, NegBin, _ = make_demand_family_envs(BeerGameParallelEnv)
    fcfg = {**ENV_BASE, "holding_cost": H_COST, "backorder_cost": B_COST}
    envs = make_heldout_family_envs(AR1, NegBin, fcfg, ar1_rhos=ar1_rhos, nb_disps=nb_disps, mu=mu)
    print(f"\n  held-out-FAMILY eval (distributional shift, mean~{mu:g}, {episodes} eps/family, CRN)")
    print(f"    {'family':<16}{'DRACO':>10}{'Optimal*':>10}{'Adaptive':>10}{'vs Opt':>9}  *comparator")
    out = {}
    for name, env in envs.items():
        cfg = env.config                                  # read the family params off the env
        fam = cfg.get("family")
        pol = DracoV4Policy(ckpt, env, ablate=False)
        d = [run_episode(pol, env, HELDOUT_SEED_BASE + e)["cost"] for e in range(episodes)]
        if fam == "ar1":                                  # MMFE/AR(1)-optimal at retailer + adaptive upstream
            opt = make_ar1_rung(mu=cfg.get("ar1_mu", mu), rho=cfg.get("ar1_rho", 0.0),
                                sigma=cfg.get("ar1_sigma", 3.0), h=H_COST, b=B_COST)
            opt_name = "AR1-opt"
        else:                                             # NegBin: Gamma-Poisson (its exact model) at retailer
            opt = make_bayes_rung(h=H_COST, b=B_COST, prior_mean=mu)
            opt_name = "Bayes"
        oz = [bl_rollout(env, opt, HELDOUT_SEED_BASE + e)[0] for e in range(episodes)]
        az = [bl_rollout(env, AdaptiveForecastPolicy(h=H_COST, b=B_COST),
                         HELDOUT_SEED_BASE + e)[0] for e in range(episodes)]
        dC, oC, aC = float(np.mean(d)), float(np.mean(oz)), float(np.mean(az))
        vs = 100.0 * (oC - dC) / max(1e-6, oC)
        out[name] = {"draco": dC, "optimal": oC, "optimal_kind": opt_name, "adaptive": aC, "vs_opt_pct": vs}
        print(f"    {name:<16}{dC:>10.1f}{oC:>10.1f}{aC:>10.1f}{vs:>8.1f}%  {opt_name}")
    print("    (*Optimal = AR1-opt (MMFE) for AR(1), Bayes (Gamma-Poisson) for NegBin. vs Opt>0 = DRACO")
    print("     below the family's proper optimum; on AR(1) compare to AR1-opt, NOT Bayes.)")
    return out


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
    ap.add_argument("--refs-json", default="results/baselines_regime_v2.json",
                    help="four-rung refs from `python scripts/baselines.py regime` (single source of "
                         "truth; derives BAR/CEILING/Bayes unless --bar/--ceiling override them)")
    ap.add_argument("--bar", type=float, default=None, help="override fixed-policy BAR (else from --refs-json)")
    ap.add_argument("--ceiling", type=float, default=None, help="override per-lambda CEILING (else from --refs-json)")
    ap.add_argument("--regime-episodes", type=int, default=20, help="episodes per held-out lambda")
    ap.add_argument("--dump-c1", default=None, metavar="DIR",
                    help="PRODUCER mode: write per-seed {lambda: cost} to DIR/seed{S}.json (the input to "
                         "c1_stats), then exit. Run once per Phase-2 checkpoint.")
    ap.add_argument("--dump-c1-episodes", type=int, default=200, help="episodes/lambda for --dump-c1")
    ap.add_argument("--seed", type=int, default=None, help="seed label for --dump-c1 (default: checkpoint's seed)")
    ap.add_argument("--messages", action="store_true", help="run the Study-2 message analysis")
    ap.add_argument("--full", action="store_true",
                    help="generate EVERYTHING: per-stage OM dashboard, belief calibration+MI, "
                         "adaptation speed, message responsiveness, comm decomposition (all scenarios)")
    ap.add_argument("--diag-episodes", type=int, default=20, help="episodes for the belief diagnostics")
    ap.add_argument("--families", action="store_true",
                    help="held-out-FAMILY eval (M2.4): AR(1)/NegBin distributional-shift tests vs the "
                         "family-aware Bayes/Adaptive rungs")
    ap.add_argument("--family-episodes", type=int, default=50)
    ap.add_argument("--family-mu", type=float, default=12.0, help="held mean demand for the family tests")
    ap.add_argument("--ar1-rhos", nargs="+", type=float, default=[0.0, 0.3, 0.6, 0.9],
                    help="AR(1) autocorrelations to test (the comm-value-vs-rho axis for Study 3)")
    ap.add_argument("--bullwhip", action="store_true",
                    help="bullwhip comparison (review #13): order-variance amplification DRACO vs static vs Bayes")
    ap.add_argument("--bullwhip-episodes", type=int, default=40)
    ap.add_argument("--action-diag", action="store_true",
                    help="action saturation diagnostics (review #11): clip@0 / at-cap / desired-order distribution")
    args = ap.parse_args()

    ckpt = torch.load(args.ckpt, map_location=DEVICE, weights_only=False)
    cfg = ckpt.get("config", {}).get("agent", {})

    # ---- PRODUCER mode: just dump per-seed per-lambda costs for c1_stats, then exit ----
    if args.dump_c1:
        dump_c1(ckpt, args.dump_c1_episodes, args.dump_c1, seed=args.seed)
        return

    # ---- resolve C1 references: explicit CLI > --refs-json > hardcoded fallback ----
    _root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    refs_path = args.refs_json if os.path.isabs(args.refs_json) else os.path.join(_root, args.refs_json)
    bar, ceiling, bayes = args.bar, args.ceiling, None
    if os.path.exists(refs_path):
        try:
            from scripts.c1_stats import load_rungs, mean_refs
            _m = mean_refs(load_rungs(refs_path), HELDOUT_LAMBDAS)
            if bar is None:
                bar = _m.get("BAR_static")
            if ceiling is None:
                ceiling = _m.get("Oracle")
            bayes = _m.get("Bayes")
            print(f"  C1 refs <- {os.path.basename(refs_path)}: "
                  f"BAR={bar} Oracle={ceiling} Bayes={bayes}")
        except Exception as e:
            print(f"  (refs-json load failed: {type(e).__name__}: {e}; using CLI/defaults)")
    if bar is None:
        bar = 4726.0
    if ceiling is None:
        ceiling = 2202.0

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
    regime_uncertainty(ckpt, args.regime_episodes, bar, ceiling, bayes=bayes)

    # ---- held-out-family eval (M2.4, distributional shift; AR(1)-rho axis for Study 3) ----
    if args.families or args.full:
        heldout_family_eval(ckpt, episodes=args.family_episodes, mu=args.family_mu,
                            ar1_rhos=tuple(args.ar1_rhos))

    # ---- bullwhip comparison (review #13: the 'adapt without bullwhip' S2 story) ----
    if args.bullwhip or args.full:
        bullwhip_comparison(ckpt, episodes=args.bullwhip_episodes, refs_json=refs_path)

    # ---- action saturation diagnostics (review #11) ----
    if args.action_diag or args.full:
        action_diagnostics(ckpt, episodes=max(5, args.bullwhip_episodes // 2))

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