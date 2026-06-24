import io
import os
import sys
import csv
import random
from collections import deque

import hydra
import numpy as np
import torch
import wandb
from omegaconf import DictConfig, OmegaConf, open_dict
from torch.distributions import Normal


def _torch_save(obj, path, _retries=6, _delay=5):
    import time
    os.makedirs(os.path.dirname(path), exist_ok=True)
    buf = io.BytesIO(); torch.save(obj, buf); data = buf.getvalue()
    for attempt in range(_retries):
        try:
            with open(path, "wb") as f:
                f.write(data)
            return
        except PermissionError:
            if attempt < _retries - 1:
                time.sleep(_delay)
            else:
                raise


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agents.draco_v4 import (
    ADJ, DemandRandomizedBeerGame, make_encoder, make_actor,
    BaseStockActor, MessageHead, DistributionalCritic, DRACOTrainerV4, DRACORolloutBuffer,
)
from agents.heldout_eval import make_heldout_envs, run_heldout_eval, HELDOUT_LAMBDAS  # held-out-lambda eval (C1 gate)
from agents.topologies import get_adj                                 # Study-2 comm topology selector
from envs.beer_game_env import BeerGameParallelEnv


def set_global_seeds(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    base_seed = cfg.get("seed", 1000)
    set_global_seeds(base_seed)
    print("[draco-v4] booting...", flush=True)

    run = wandb.init(project="BeerGame_Research", name=cfg.agent.algorithm)
    wandb.define_metric("Avg_Cost_50", summary="min")
    wandb.define_metric("Avg_Cost_500", summary="last")
    wandb.config.update(OmegaConf.to_container(cfg, resolve=True), allow_val_change=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    env_cfg = OmegaConf.to_container(cfg.env, resolve=True)
    _dr_kw = dict(
        lam_lo=cfg.agent.get("dr_lambda_lo", 4.0),
        lam_hi=cfg.agent.get("dr_lambda_hi", 16.0),
        p_shift=cfg.agent.get("dr_p_shift", 0.5),
        shift_scale=cfg.agent.get("dr_shift_scale", 2.0),
    )
    env = DemandRandomizedBeerGame(env_cfg, **_dr_kw)
    obs, _ = env.reset(seed=base_seed)

    eval_env_poisson = BeerGameParallelEnv({**env_cfg, "demand_type": "poisson"})
    eval_ood_type = cfg.agent.get("ood_eval_type", "black_swan")
    eval_env_ood = BeerGameParallelEnv({**env_cfg, "demand_type": eval_ood_type})
    eval_every = cfg.agent.get("eval_every", 50)
    eval_episodes = cfg.agent.get("eval_episodes", 5)

    # --- held-out-lambda eval (the C1 gate): score DRACO on stationary UNKNOWN poisson
    #     regimes against the deployable fixed-policy BAR and the per-lambda CEILING from
    #     `python baselines.py regime`. Watch Eval_lambda/Gap_Recovered: >=0 beats the bar
    #     (C1 exists), 1.0 == oracle, <0 == no regime inference. Envs are built ONCE here. ---
    heldout_lams   = list(cfg.agent.get("heldout_lambdas", HELDOUT_LAMBDAS))  # val split (Phase1) vs test split (Phase2)
    heldout_envs   = make_heldout_envs(DemandRandomizedBeerGame, env_cfg, heldout_lams)
    heldout_every  = cfg.agent.get("heldout_every", 200)
    heldout_eps    = cfg.agent.get("heldout_episodes", 20)
    heldout_fixed  = cfg.agent.get("heldout_fixed_ref", 4726.0)   # BAR     from `baselines.py regime`
    heldout_oracle = cfg.agent.get("heldout_oracle_ref", 2202.0)  # CEILING from `baselines.py regime`

    run_dir = os.path.join(_ROOT, "weights_draco", f"run_dracov4_{run.id}")
    os.makedirs(run_dir, exist_ok=True)

    agents = list(env.possible_agents)
    N = len(agents)
    local_dim = env.observation_space("retailer").shape[0]
    gdim = len(env.get_global_state())
    max_order = env.max_order
    horizon = int(env.horizon)

    hidden = cfg.agent.hidden_dim
    z_dim = cfg.agent.get("z_dim", 8)
    msg_dim = cfg.agent.get("msg_dim", 4)
    n_quant = cfg.agent.get("n_quantiles", 32)
    encoder_type = cfg.agent.get("encoder_type", "gru")       # recurrent belief by default
    belief_sample = bool(cfg.agent.get("belief_sample", False))
    adj = get_adj(cfg.agent.get("comm_topology", "neighbor")).to(device)   # Study-2 topology selector

    # --- guard: the CRAFT positional-encoding window must cover the whole episode
    #     (the update encodes the full T=horizon sequence at once). ---
    safe_max_len = max(int(cfg.agent.get("craft_max_len", 64)), horizon + 1)
    with open_dict(cfg.agent):
        cfg.agent.craft_max_len = safe_max_len
    craft_max_len = safe_max_len

    s_bias_init = cfg.agent.get("s_bias_init", 40.0)
    s_logstd_init = cfg.agent.get("s_logstd_init", 0.7)
    use_comm = cfg.agent.get("use_comm", True)
    msg_mode = cfg.agent.get("msg_mode", "learned")   # "learned" DIAL vector | "dhat" broadcast (Phase 3)
    use_context = cfg.agent.get("use_context", True)

    # --- modules (B encoder | A actors | C msg heads | D belief-conditioned critic) ---
    # NOTE: the head selector is `actor_head` ('structured' | 'mlp'); `structured_head`
    # in the yaml is NOT read by anything. Set actor_head=structured for the primary head.
    actor_head = cfg.agent.get("actor_head", "structured")
    encoder = make_encoder(encoder_type, local_dim, msg_dim, z_dim, cfg.agent).to(device)
    actors = [make_actor(actor_head, local_dim, z_dim, msg_dim, hidden, max_order, cfg.agent).to(device)
              for _ in range(N)]
    msg_heads = [MessageHead(local_dim, z_dim, msg_dim, hidden).to(device) for _ in range(N)]
    # critic takes the global state AND the flattened detached belief (N*z_dim).
    critic = DistributionalCritic(gdim, N * z_dim, hidden, N, n_quant).to(device)
    trainer = DRACOTrainerV4(encoder, actors, msg_heads, critic, cfg.agent, device, N, adj)

    step = cfg.agent.lr_scheduler_step
    gamma_s = cfg.agent.lr_scheduler_gamma
    schedulers = [
        torch.optim.lr_scheduler.StepLR(trainer.policy_opt, step, gamma_s),
        torch.optim.lr_scheduler.StepLR(trainer.critic_opt, step, gamma_s),
        torch.optim.lr_scheduler.StepLR(trainer.enc_opt, step, gamma_s),
    ]

    warm_up = cfg.agent.get("warm_up_episodes", 500)
    patience = cfg.agent.get("patience", 3000)
    trace_every = cfg.agent.get("trace_every", 0)
    cost_hist, cost_hist_500 = deque(maxlen=50), deque(maxlen=500)
    best, since_imp = float("inf"), 0

    print(f"[draco-v4] built: encoder={encoder_type}, actor_head={actor_head}, {N} actors+msg-heads, "
          f"z={z_dim}, msg={msg_dim}, quantiles={n_quant}, belief->critic={trainer.critic_uses_belief}, "
          f"comm={use_comm}. Starting loop.", flush=True)

    @torch.no_grad()
    def run_episode(ep, target_env, collect, deterministic=False, trace_rows=None):
        obs_local, _ = target_env.reset(seed=base_seed + ep)
        cur = target_env.possible_agents
        m_buf = torch.zeros(N, msg_dim, device=device)            # previous-step OUTGOING messages (one-step delay)
        obs_hist, msg_hist = [], []
        buf = DRACORolloutBuffer()
        ep_cost = 0.0
        ep_costs = {a: 0.0 for a in cur}
        msgs_log = []
        s_sum = order_sum = 0.0
        nstep = 0
        while True:
            o_arr = np.stack([obs_local[a] for a in cur])
            o_t = torch.tensor(o_arr, dtype=torch.float32, device=device)        # [N,od]
            g_t = torch.tensor(target_env.get_global_state(), dtype=torch.float32, device=device).view(-1)

            incoming = adj @ m_buf                                                # [N,msg] routed, delayed
            if not use_comm:
                incoming = torch.zeros_like(incoming)

            # belief via prefix re-encode (uniform across GRU/CRAFT; action-free, causal)
            obs_hist.append(o_t); msg_hist.append(incoming)
            obs_seq = torch.stack(obs_hist[-craft_max_len:])                      # [t,N,od]
            msg_seq = torch.stack(msg_hist[-craft_max_len:])                      # [t,N,msg]
            mu, ls, _ = encoder.forward_sequence(obs_seq, msg_seq)
            z_t = mu[-1]                                                          # [N,z] posterior mean
            if belief_sample:
                z_t = z_t + torch.randn_like(z_t) * ls[-1].exp()                 # BAMDP: act on a sample
            z_act = z_t if use_context else torch.zeros_like(z_t)

            # base-stock action S (the only stochastic policy output)
            S = torch.zeros(N, 1, device=device)
            logp = torch.zeros(N, 1, device=device)
            for i in range(N):
                s_mu, s_std = actors[i](o_t[i:i + 1], z_act[i:i + 1], incoming[i:i + 1])
                S_i = s_mu if deterministic else Normal(s_mu, s_std).rsample()
                logp[i] = Normal(s_mu, s_std).log_prob(S_i).sum()
                S[i] = S_i                                                       # <-- REQUIRED: write the action

            # lightweight per-agent diagnostic (training rollouts only; S is now populated)
            #if collect and ep % 20 == 0 and target_env.current_step == 10:
            #    _names = ["Retailer", "Wholesaler", "Distributor", "Manufacturer"]
            #    print(f"--- EP {ep} DIAG (step 10) ---", flush=True)
            #    for i, name in enumerate(_names):
            #        print(f"  {name:12s} S={S[i, 0].item():7.2f}  z_mean={z_act[i].mean().item():+.3f}", flush=True)

            # messages: learned DIAL vector, OR (Phase 3) broadcast the detached demand belief d_hat
            m_out = torch.zeros(N, msg_dim, device=device)
            if use_comm:
                if msg_mode == "dhat":                                            # d_hat broadcast -> interpretable, msg_dim must be 1
                    for i in range(N):
                        m_out[i] = actors[i].demand_estimate(z_t[i:i + 1]).reshape(-1)
                else:
                    for i in range(N):
                        m_out[i] = msg_heads[i](o_t[i:i + 1], z_t[i:i + 1]).squeeze(0)

            order, IP = BaseStockActor.order_from_S(S, o_t, max_order)            # [N,1]
            frac = (order / max_order).clamp(0.0, 1.0)
            acts = {a: [float(frac[i, 0].item())] for i, a in enumerate(cur)}
            s_sum += float(S.mean().item()); order_sum += float(order.mean().item()); nstep += 1

            if trace_rows is not None:
                for i, a in enumerate(cur):
                    trace_rows.append({
                        "ep": ep, "t": target_env.current_step, "agent": a,
                        "inv": float(o_t[i, 0]), "backlog": float(o_t[i, 1]),
                        "on_order": float(o_t[i, 2]), "last_demand": float(o_t[i, 3]),
                        "IP": float(IP[i, 0]), "S_target": float(S[i, 0]), "order": float(order[i, 0]),
                        **{f"z{j}": float(z_t[i, j]) for j in range(z_dim)},
                        **{f"msg_in{j}": float(incoming[i, j]) for j in range(msg_dim)},
                        **{f"msg_out{j}": float(m_out[i, j]) for j in range(msg_dim)},
                    })

            next_obs, rewards, terms, truncs, infos = target_env.step(acts)
            cost_vec = torch.tensor([float(infos[a]["local_cost"]) for a in cur],
                                    dtype=torch.float32, device=device)          # [N] raw per-agent cost
            raw_cost = float(cost_vec.sum().item())
            for i, a in enumerate(cur):
                ep_costs[a] += float(cost_vec[i].item())
            ep_cost += raw_cost
            done = any(terms.values()) or any(truncs.values())
            term = torch.tensor([1.0 if done else 0.0], device=device)
            demand_tgt = torch.tensor([[float(infos[a]["training_targets"]["demand"])] for a in cur],
                                      dtype=torch.float32, device=device)         # [N,1] realized-this-step demand
            if collect:
                buf.push(obs=o_t, g=g_t.view(-1), msg_in=incoming.detach(),
                         S_act=S.detach(), logp=logp.detach(), cost=cost_vec,
                         done=term, demand_tgt=demand_tgt)
            msgs_log.append(m_out.detach().cpu().numpy())
            m_buf = m_out.detach()
            obs_local = next_obs
            if done:
                break
        return buf, ep_cost, ep_costs, msgs_log, (s_sum / max(1, nstep), order_sum / max(1, nstep))

    batch_eps = cfg.agent.get("batch_episodes", 32)
    episode_buffers = []
    a_loss = c_loss = e_loss = 0.0
    for ep in range(cfg.total_episodes):
        train_this = ep >= warm_up
        buf, ep_cost, ep_costs, msgs_log, (s_mean, order_mean) = run_episode(
            ep, target_env=env, collect=train_this, deterministic=False)
        if train_this and len(buf) > 0:
            episode_buffers.append(buf)
            if len(episode_buffers) >= batch_eps:
                a_loss, c_loss, e_loss = trainer.update(episode_buffers)
                for s in schedulers:
                    s.step()
                episode_buffers = []

        # ---- per-episode logging (everything below is INSIDE the for-loop) ----
        cost_hist.append(ep_cost); cost_hist_500.append(ep_cost)
        avg = sum(cost_hist) / len(cost_hist)
        log = {"Cost": ep_cost, "Avg_Cost_50": avg, "Avg_Cost_500": sum(cost_hist_500) / len(cost_hist_500),
               "Actor_Loss": a_loss, "Critic_Loss": c_loss, "Encoder_Loss": e_loss,
               "Diag/S_mean": s_mean, "Diag/Order_mean": order_mean}
        for a, c in ep_costs.items():
            log[f"Cost/{a}"] = c
        if msgs_log:
            arr = np.concatenate(msgs_log, axis=0)
            log["Comm/Msg_Mean_Abs"] = float(np.abs(arr).mean())
            log["Comm/Msg_Std"] = float(arr.std())

        # periodic held-out eval (poisson + OOD), logging per-regime S/order so you can
        # SEE whether the policy raises S under the OOD regime (the level-tracking test).
        if ep > warm_up and ep % eval_every == 0:
            p_costs, o_costs = [], []
            p_s_means, o_s_means = [], []
            p_ord_means, o_ord_means = [], []
            for e_idx in range(eval_episodes):
                eval_ep_idx = 100000 + e_idx                       # held-out seed space
                _, p_c, _, _, (p_s, p_ord) = run_episode(eval_ep_idx, eval_env_poisson, collect=False, deterministic=True)
                _, o_c, _, _, (o_s, o_ord) = run_episode(eval_ep_idx, eval_env_ood, collect=False, deterministic=True)
                p_costs.append(p_c); o_costs.append(o_c)
                p_s_means.append(p_s); o_s_means.append(o_s)
                p_ord_means.append(p_ord); o_ord_means.append(o_ord)
            mean_p = sum(p_costs) / eval_episodes
            mean_o = sum(o_costs) / eval_episodes
            log["Eval/Poisson_Cost"] = mean_p
            log[f"Eval/{eval_ood_type}_Cost"] = mean_o
            log["Eval/Generalization_Gap"] = mean_o - mean_p
            log["Eval/Poisson_S_mean"] = sum(p_s_means) / eval_episodes
            log[f"Eval/{eval_ood_type}_S_mean"] = sum(o_s_means) / eval_episodes
            log["Eval/Poisson_Order_mean"] = sum(p_ord_means) / eval_episodes
            log[f"Eval/{eval_ood_type}_Order_mean"] = sum(o_ord_means) / eval_episodes

        # held-out-lambda eval (C1 gate): reuses run_episode (deterministic) on the
        # stationary unknown-regime envs; adds Eval_lambda/* incl. Gap_Recovered to `log`.
        if ep > warm_up and ep % heldout_every == 0:
            log.update(run_heldout_eval(
                run_episode, heldout_envs, base_seed, heldout_eps,
                fixed_ref=heldout_fixed, oracle_ref=heldout_oracle))

        wandb.log(log)

        # checkpoint on improvement (every episode)
        if ep == warm_up:
            best, since_imp = float("inf"), 0
        if avg < best and len(cost_hist) == 50:
            best, since_imp = avg, 0
            if ep >= warm_up:
                _torch_save({"encoder": encoder.state_dict(),
                             "actors": [a.state_dict() for a in actors],
                             "msg_heads": [m.state_dict() for m in msg_heads],
                             "critic": critic.state_dict(),
                             "config": OmegaConf.to_container(cfg, resolve=True),
                             "episode": ep, "best_avg_cost": best},
                            os.path.join(run_dir, "draco_checkpoint_best.pt"))
        else:
            if ep >= warm_up:
                since_imp += 1

        if trace_every and ep > warm_up and ep % trace_every == 0:
            rows = []
            run_episode(ep, target_env=env, collect=False, deterministic=True, trace_rows=rows)
            if rows:
                path = os.path.join(run_dir, f"trace_ep{ep}.csv")
                with open(path, "w", newline="") as f:
                    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

        if ep > warm_up and since_imp >= patience:
            print(f"[draco-v4] early stop at ep {ep}.", flush=True)
            break
        if ep % 10 == 0 or ep < 3:
            print(f"Ep {ep} | Cost {ep_cost:.1f} | 50-avg {avg:.1f} | "
                  f"best {best if best != float('inf') else 0:.1f}", flush=True)

    wandb.finish()


if __name__ == "__main__":
    main()