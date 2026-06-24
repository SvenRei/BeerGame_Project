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
        self.m_buf = m_out
        return {a: float(frac[i, 0].item()) for i, a in enumerate(AGENTS)}


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
    tr_dhat, tr_msg, tr_cust = [], [], []          # per-step traces (message analysis)
    while True:
        acts = policy.act(obs)
        for a in AGENTS:
            orders[a].append(int(np.floor(np.clip(acts[a], 0, 1) * env.max_order + 0.5)))
        cust = float(env.current_incoming_order["retailer"])   # true customer demand (~ regime)
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
    out = {
        "cost": tot_cost,
        "bw_overall": _safe_ratio(np.var(orders["manufacturer"]), np.var(demand["retailer"])),
        "avg_inv": float(all_inv.mean()),
        "avg_back": float(all_back.mean()),
        "ret_service_alpha": float(np.mean(np.array(back["retailer"]) == 0)),
        "fill_beta": _safe_ratio(fill_m["retailer"], fill_d["retailer"]),
        "jitter": float(np.mean([np.mean(np.abs(np.diff(orders[a]))) if len(orders[a]) > 1 else 0.0
                                 for a in AGENTS])),
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
        sat = float(np.mean(np.abs(np.tanh(ch)) > 0.9))
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
    for scenario in SCENARIOS:
        env = BeerGameParallelEnv({**ENV_BASE, "demand_type": scenario})
        eps = evaluate(DracoV4Policy(ckpt, env, ablate=False), env, args.episodes)
        costs = np.array([e["cost"] for e in eps])
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

    # ---- regime-uncertainty (C1) ----
    regime_uncertainty(ckpt, args.regime_episodes, args.bar, args.ceiling)

    # ---- message analysis (Study 2) ----
    if args.messages:
        message_analysis(ckpt, episodes=min(40, args.episodes))
    print()


if __name__ == "__main__":
    main()