"""
test_draco_v4.py -- component & integration test suite for DRACO v4.

Run from the repo root (the dir that contains agents/ and envs/):

    python test_draco_v4.py

Requires only torch + pettingzoo (your normal training environment). No W&B, no
Hydra, no GPU needed -- it runs on CPU in a few seconds. The point is to prove that
every head, encoder, the critic, the value-normalizer, the gradient paths, and the
full obs->S->order pipeline are wired correctly BEFORE you spend RunPod hours.

Exit code is 0 iff every test passes.
"""
import os
import sys
import math
import traceback

import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Normal

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root (parent of test/) on path

from agents.draco_v4 import (
    ADJ, DemandRandomizedBeerGame, make_encoder, make_actor,
    BaseStockActor, BaseStockActorStructured, MessageHead,
    DistributionalCritic, ValueCritic, RunningNorm,
    DRACOTrainerV4, DRACORolloutBuffer,
)
from envs.beer_game_env import BeerGameParallelEnv

torch.manual_seed(0)
np.random.seed(0)

# ----------------------------------------------------------------------------- tiny test runner
_PASS, _FAIL, _FAILED = 0, 0, []


def run(name, fn):
    global _PASS, _FAIL
    try:
        fn()
        _PASS += 1
        print(f"  [PASS] {name}")
    except AssertionError as e:
        _FAIL += 1
        _FAILED.append(name)
        print(f"  [FAIL] {name}  -- {e}")
    except Exception as e:
        _FAIL += 1
        _FAILED.append(name)
        print(f"  [FAIL] {name}  -- {type(e).__name__}: {e}")
        traceback.print_exc()


def section(t):
    print(f"\n=== {t} ===")


# ----------------------------------------------------------------------------- shared fixtures
ENVC = dict(horizon=50, max_order=100, holding_cost=0.5, backorder_cost=1.0, demand_type="poisson")
N = 4
OD = 4
MSG = 4
Z = 8
HID = 32
CFG = dict(hidden_dim=HID, z_dim=Z, msg_dim=MSG, n_quantiles=1, craft_max_len=64,
           s_bias_init=40.0, s_logstd_init=0.7, lead_init=4.0, demand_init=8.0, corr_scale=6.0)
_env_for_dims = BeerGameParallelEnv(ENVC)
_env_for_dims.reset(seed=0)
GDIM = len(_env_for_dims.get_global_state())


def make_modules(actor_head="structured", encoder_type="gru", n_quant=1, extra=None):
    cfg = dict(CFG); cfg["n_quantiles"] = n_quant
    if extra:
        cfg.update(extra)
    enc = make_encoder(encoder_type, OD, MSG, Z, cfg)
    actors = [make_actor(actor_head, OD, Z, MSG, HID, 100, cfg) for _ in range(N)]
    mh = [MessageHead(OD, Z, MSG, HID) for _ in range(N)]
    crit = ValueCritic(GDIM, N * Z, HID, N, n_quant)
    trn = DRACOTrainerV4(enc, actors, mh, crit, cfg, torch.device("cpu"), N, ADJ.clone())
    return cfg, enc, actors, mh, crit, trn


def synth_buffers(n_eps=2, T=10):
    bufs = []
    for _ in range(n_eps):
        b = DRACORolloutBuffer()
        for t in range(T):
            b.push(
                obs=torch.rand(N, OD) * 30,
                g=torch.randn(GDIM),
                msg_in=torch.tanh(torch.randn(N, MSG)),
                S_act=torch.rand(N, 1) * 40 + 20,
                logp=torch.randn(N, 1),
                cost=torch.rand(N) * 10,
                done=torch.tensor([1.0 if t == T - 1 else 0.0]),
                demand_tgt=torch.rand(N, 1) * 15,
            )
        bufs.append(b)
    return bufs


# ============================================================================= A. ENVIRONMENT
def test_env():
    section("A. Environment mechanics")

    def reset_shapes():
        env = BeerGameParallelEnv(ENVC)
        obs, _ = env.reset(seed=1)
        for a in env.possible_agents:
            assert obs[a].shape == (4,), f"{a} obs shape {obs[a].shape}"
        assert len(env.get_global_state()) == GDIM
    run("reset: obs is 4 local scalars, global state stable", reset_shapes)

    def onorder_reset_invariant():
        env = BeerGameParallelEnv(ENVC)
        obs, _ = env.reset(seed=2)
        for a in env.possible_agents:
            ship = sum(env.shipment_pipelines[a].pipeline.values())
            order = sum(env.order_pipelines[a].pipeline.values())
            assert abs(obs[a][2] - (ship + order)) < 1e-6, f"{a}: on_order {obs[a][2]} != {ship+order}"
    run("reset: on_order == ship_pipe + order_pipe", onorder_reset_invariant)

    def obs_matches_internal():
        env = BeerGameParallelEnv(ENVC)
        obs, _ = env.reset(seed=3)
        for _ in range(20):
            acts = {a: [0.2] for a in env.agents}
            obs, *_ = env.step(acts)
            for a in env.possible_agents:
                assert abs(obs[a][0] - env.inventory[a]) < 1e-6
                assert abs(obs[a][1] - env.backlog[a]) < 1e-6
                assert abs(obs[a][2] - env.unfulfilled_orders[a]) < 1e-6
    run("obs slots track internal inventory/backlog/on_order every step", obs_matches_internal)

    def order_from_S_math():
        obs = torch.tensor([[12.0, 0.0, 16.0, 8.0]])           # IP = 12-0+16 = 28
        order, IP = BaseStockActor.order_from_S(torch.tensor([[40.0]]), obs, 100)
        assert abs(IP.item() - 28) < 1e-6, IP.item()
        assert abs(order.item() - 12) < 1e-6, order.item()      # 40-28
    run("order_from_S: order = clamp(S - IP)", order_from_S_math)

    def order_S_zero_is_the_bug():
        # the exact regression we hit: S=0 must give order 0 when IP>0 (no proactive order)
        obs = torch.tensor([[12.0, 0.0, 16.0, 8.0]])
        order, _ = BaseStockActor.order_from_S(torch.zeros(1, 1), obs, 100)
        assert order.item() == 0.0, f"S=0 should give order 0, got {order.item()}"
    run("order_from_S: S=0 -> order 0 (the starvation regression sentinel)", order_S_zero_is_the_bug)

    def order_saturates():
        obs = torch.tensor([[0.0, 50.0, 0.0, 8.0]])             # IP = -50
        order, _ = BaseStockActor.order_from_S(torch.tensor([[300.0]]), obs, 100)
        assert order.item() == 100.0, f"should saturate at max_order, got {order.item()}"
    run("order_from_S: saturates at max_order", order_saturates)

    def demand_regimes():
        # black_swan: ~8 before t=25, ~20 after
        env = BeerGameParallelEnv({**ENVC, "demand_type": "black_swan"})
        lo = np.mean([env._roll_stochastic_demand(t) for t in range(5, 24) for _ in range(50)])
        hi = np.mean([env._roll_stochastic_demand(t) for t in range(25, 49) for _ in range(50)])
        assert 5 < lo < 11, f"black_swan pre-shift mean {lo}"
        assert 16 < hi < 24, f"black_swan post-shift mean {hi}"
        # poisson ~8
        envp = BeerGameParallelEnv(ENVC)
        m = np.mean([envp._roll_stochastic_demand(t) for t in range(50) for _ in range(50)])
        assert 6 < m < 10, f"poisson mean {m}"
    run("demand regimes: poisson~8, black_swan steps 8->20 at t=25", demand_regimes)

    def cost_formula():
        env = BeerGameParallelEnv(ENVC)
        env.reset(seed=4)
        _, _, _, _, infos = env.step({a: [0.1] for a in env.agents})
        for a in env.possible_agents:
            expect = 0.5 * env.inventory[a] + 1.0 * env.backlog[a]
            assert abs(infos[a]["local_cost"] - expect) < 1e-6
    run("cost = holding*inv + backorder*backlog", cost_formula)

    def dr_perturbs_only_poisson():
        # DR wrapper must pass black_swan straight through (OOD benchmark untouched)
        dr = DemandRandomizedBeerGame({**ENVC, "demand_type": "black_swan"},
                                      lam_lo=6, lam_hi=20, p_shift=0.4, shift_scale=2.5)
        dr.reset(seed=5)
        hi = np.mean([dr._roll_stochastic_demand(t) for t in range(25, 49) for _ in range(50)])
        assert 16 < hi < 24, f"DR must not alter black_swan, got post-shift mean {hi}"
    run("DemandRandomizedBeerGame: leaves black_swan unaltered", dr_perturbs_only_poisson)


# ============================================================================= B. ACTOR HEADS
def test_heads():
    section("B. Actor heads")

    def both_heads_init_near_bias():
        for head in ("mlp", "structured"):
            a = make_actor(head, OD, Z, MSG, HID, 100, CFG)
            obs = torch.tensor([[12.0, 0.0, 16.0, 8.0]])
            z = torch.zeros(1, Z)
            m = torch.zeros(1, MSG)
            s_mu, s_std = a(obs, z, m)
            assert 30 < s_mu.item() < 50, f"{head} init S={s_mu.item()} (want ~40)"
            assert s_std.item() > 0
    run("both heads initialize S ~= s_bias_init (40)", both_heads_init_near_bias)

    def structured_demand_grounding():
        a = make_actor("structured", OD, Z, MSG, HID, 100, CFG)
        z0 = torch.zeros(1, Z)
        d0 = a.demand_estimate(z0).item()
        assert abs(d0 - 8.0) < 0.5, f"d_hat at z=0 should be ~demand_init=8, got {d0}"
        # if d_head learns a positive map, raising z must raise d_hat -> raise S (level tracking)
        with torch.no_grad():
            a.d_head.weight.fill_(0.5)
        d1 = a.demand_estimate(torch.ones(1, Z)).item()
        assert d1 > d0 + 2, f"d_hat must rise with z once d_head!=0: {d0}->{d1}"
        s_lo, _ = a(torch.tensor([[12., 0., 16., 8.]]), z0, torch.zeros(1, MSG))
        s_hi, _ = a(torch.tensor([[12., 0., 16., 8.]]), torch.ones(1, Z), torch.zeros(1, MSG))
        assert s_hi.item() > s_lo.item() + 2, f"S must track d_hat: {s_lo.item()}->{s_hi.item()}"
    run("structured head: S = lead*d_hat + safety tracks the demand estimate", structured_demand_grounding)

    def corr_is_bounded():
        a = make_actor("structured", OD, Z, MSG, HID, 100, CFG)
        with torch.no_grad():
            for p in a.corr_net.parameters():
                p.mul_(0).add_(5.0)                    # push corr_net to extreme
        a(torch.rand(8, OD) * 50, torch.randn(8, Z), torch.randn(8, MSG))
        assert a._last_corr.abs().max().item() <= CFG["corr_scale"] + 1e-4, \
            f"corr must be bounded by corr_scale, max |corr|={a._last_corr.abs().max().item()}"
    run("structured head: corr bounded to +/- corr_scale", corr_is_bounded)

    def head_batches():
        for head in ("mlp", "structured"):
            a = make_actor(head, OD, Z, MSG, HID, 100, CFG)
            s_mu, s_std = a(torch.rand(7, OD) * 30, torch.randn(7, Z), torch.randn(7, MSG))
            assert s_mu.shape == (7, 1), f"{head} s_mu shape {s_mu.shape}"
            assert (s_mu >= 0).all(), f"{head} S must be >=0 (softplus)"
    run("both heads: batched forward gives [B,1] non-negative S", head_batches)

    def message_head_bounded():
        mh = MessageHead(OD, Z, MSG, HID)
        out = mh(torch.rand(5, OD) * 100, torch.randn(5, Z) * 10)
        assert out.shape == (5, MSG)
        assert out.abs().max().item() <= 1.0 + 1e-6, "messages must be tanh-bounded to [-1,1]"
    run("MessageHead: output is [B,msg] bounded to [-1,1]", message_head_bounded)


# ============================================================================= C. ENCODERS
def test_encoders():
    section("C. Encoders (belief)")

    def shapes_and_range():
        for kind in ("gru", "craft"):
            enc = make_encoder(kind, OD, MSG, Z, CFG)
            T = 12
            mu, ls, pred = enc.forward_sequence(torch.rand(T, N, OD) * 30, torch.tanh(torch.randn(T, N, MSG)))
            assert mu.shape == (T, N, Z), f"{kind} mu {mu.shape}"
            assert ls.shape == (T, N, Z), f"{kind} logstd {ls.shape}"
            assert pred.shape == (T, N, 1), f"{kind} pred {pred.shape}"
            assert ls.min().item() >= -5.01 and ls.max().item() <= 2.01, f"{kind} logstd not clamped"
    run("both encoders: forward_sequence shapes + logstd clamp", shapes_and_range)

    def causality():
        # z[t] must not depend on inputs at t' > t. Perturb the LAST step, earlier z must not move.
        for kind in ("gru", "craft"):
            enc = make_encoder(kind, OD, MSG, Z, CFG)
            T = 10
            obs = torch.rand(T, N, OD) * 30
            msg = torch.tanh(torch.randn(T, N, MSG))
            mu1, _, _ = enc.forward_sequence(obs, msg)
            obs2 = obs.clone(); obs2[-1] += 1000.0                 # corrupt only the final step
            mu2, _, _ = enc.forward_sequence(obs2, msg)
            diff_before = (mu1[:-1] - mu2[:-1]).abs().max().item()
            assert diff_before < 1e-4, f"{kind} NOT causal: changing t=T-1 moved earlier z by {diff_before}"
    run("both encoders: causal (future input cannot change past belief)", causality)


# ============================================================================= D. CRITIC + NORM
def test_critic_norm():
    section("D. Critic + value normalizer")

    def critic_shapes():
        c = ValueCritic(GDIM, N * Z, HID, N, 1)
        out = c(torch.randn(6, GDIM), torch.randn(6, N, Z))
        assert out.shape == (6, N), f"critic out {out.shape}"
        out2 = c(torch.randn(6, GDIM), torch.randn(6, N * Z))    # accepts flattened belief too
        assert out2.shape == (6, N)
    run("critic: per-agent scalar value [B,N], accepts [B,N,z] or [B,N*z]", critic_shapes)

    def alias_is_scalar():
        assert DistributionalCritic is ValueCritic, "train script imports DistributionalCritic; must alias ValueCritic"
        c = DistributionalCritic(GDIM, N * Z, HID, N, 32)         # n_quant ignored now
        assert c(torch.randn(3, GDIM), torch.randn(3, N, Z)).shape == (3, N)
    run("DistributionalCritic alias resolves to scalar ValueCritic", alias_is_scalar)

    def running_norm_converges():
        rn = RunningNorm()
        g = torch.randn(20000) * 37.0 - 800.0                    # mean -800, std 37
        for i in range(0, 20000, 500):
            rn.update(g[i:i + 500])
        assert abs(rn.mean - (-800.0)) < 5.0, f"RunningNorm mean {rn.mean}"
        assert abs(rn.std - 37.0) < 3.0, f"RunningNorm std {rn.std}"
    run("RunningNorm: recovers mean/std of the value target (PopArt-lite)", running_norm_converges)


# ============================================================================= E. TRAINER / GRADIENTS
def test_trainer():
    section("E. Trainer & gradient flow")

    def update_runs_finite():
        _, _, _, _, _, trn = make_modules("structured", "gru")
        a, c, e = trn.update(synth_buffers())
        assert all(math.isfinite(x) for x in (a, c, e)), f"non-finite losses {(a, c, e)}"
    run("trainer.update returns finite (actor, critic, encoder) losses", update_runs_finite)

    def all_groups_train():
        cfg, enc, actors, mh, crit, trn = make_modules("structured", "gru", extra={"use_comm": True})
        snap = lambda mods: [p.detach().clone() for m in mods for p in m.parameters()]
        b_enc, b_cri = snap([enc]), snap([crit])
        b_act, b_msg = snap(actors), snap(mh)
        trn.update(synth_buffers())
        moved = lambda before, mods: any(
            not torch.equal(x, p) for x, p in zip(before, (q for m in mods for q in m.parameters())))
        assert moved(b_enc, [enc]), "encoder did not update (ELBO path dead)"
        assert moved(b_cri, [crit]), "critic did not update"
        assert moved(b_act, actors), "actors did not update (PPO path dead)"
        assert moved(b_msg, mh), "msg heads did not update (DIAL path dead)"
    run("trainer.update trains encoder + critic + actors + msg-heads", all_groups_train)

    def comm_off_freezes_msg_heads():
        cfg, enc, actors, mh, crit, trn = make_modules("structured", "gru", extra={"use_comm": False})
        before = [p.detach().clone() for m in mh for p in m.parameters()]
        trn.update(synth_buffers())
        moved = any(not torch.equal(x, p) for x, p in zip(before, (q for m in mh for q in m.parameters())))
        assert not moved, "with use_comm=False the message heads must NOT receive gradient"
    run("use_comm=False: message heads receive no gradient (DIAL gated)", comm_off_freezes_msg_heads)

    def belief_is_detached():
        _, enc, _, _, crit, trn = make_modules("structured", "gru")
        bufs = synth_buffers()
        d = dict(obs=torch.stack(bufs[0].obs), msg_in=torch.stack(bufs[0].msg_in), g=torch.stack(bufs[0].g))
        z = trn._encode_belief(d["obs"], d["msg_in"])
        assert not z.requires_grad, "belief fed to policy/critic must be detached"
        for p in enc.parameters():
            p.grad = None
        zc = trn._zero_belief(z)
        loss = trn.critic(d["g"], zc).pow(2).mean()
        loss.backward()
        assert all(p.grad is None for p in enc.parameters()), "critic loss leaked gradient into the encoder"
    run("belief detached: critic loss does not corrupt the encoder", belief_is_detached)

    def value_denorm():
        _, _, _, _, crit, trn = make_modules("structured", "gru")
        trn.ret_norm.mean, trn.ret_norm.var = -500.0, 100.0 ** 2  # std 100
        g = torch.randn(4, GDIM); zc = torch.zeros(4, N * Z)
        raw = trn.critic(g, zc)
        deno = trn._critic_value(g, zc)
        assert torch.allclose(deno, raw * 100.0 + (-500.0), atol=1e-4), "value de-normalization is wrong"
    run("value head de-normalizes via RunningNorm (raw*std+mean)", value_denorm)

    def risk_path_runs():
        _, _, _, _, _, trn = make_modules("structured", "gru", extra={"risk_eta": 0.2, "cvar_alpha": 0.3})
        a, c, e = trn.update(synth_buffers(n_eps=4))
        assert all(math.isfinite(x) for x in (a, c, e))
    run("risk_eta>0: empirical-CVaR advantage reweighting runs (scalar critic)", risk_path_runs)


# ============================================================================= F. ACTION PIPELINE (end-to-end)
def test_pipeline():
    section("F. Full obs->S->order pipeline (the regression we fixed)")

    def mini_rollout(env, actors, enc, mh, use_comm=True, deterministic=True):
        obs, _ = env.reset(seed=0)
        cur = env.possible_agents
        m_buf = torch.zeros(N, MSG)
        oh, mhist = [], []
        S_list, ord_list, cost = [], [], 0.0
        while True:
            o_t = torch.tensor(np.stack([obs[a] for a in cur]), dtype=torch.float32)
            inc = (ADJ @ m_buf) if use_comm else torch.zeros(N, MSG)
            oh.append(o_t); mhist.append(inc)
            mu, _, _ = enc.forward_sequence(torch.stack(oh), torch.stack(mhist))
            z_t = mu[-1]
            S = torch.zeros(N, 1)
            for i in range(N):
                s_mu, s_std = actors[i](o_t[i:i + 1], z_t[i:i + 1], inc[i:i + 1])
                S[i] = s_mu if deterministic else Normal(s_mu, s_std).rsample()
            order, IP = BaseStockActor.order_from_S(S, o_t, env.max_order)
            frac = (order / env.max_order).clamp(0, 1)
            acts = {a: [float(frac[i, 0])] for i, a in enumerate(cur)}
            m_out = torch.zeros(N, MSG)
            if use_comm:
                for i in range(N):
                    m_out[i] = mh[i](o_t[i:i + 1], z_t[i:i + 1]).squeeze(0)
            S_list.append(float(S.mean())); ord_list.append(float(order.mean()))
            obs, _, _, trunc, info = env.step(acts)
            cost += info[cur[0]]["supply_chain_cost"]
            m_buf = m_out.detach()
            if any(trunc.values()):
                break
        return np.array(S_list), np.array(ord_list), cost

    def s_is_populated():
        _, enc, actors, mh, _, _ = make_modules("structured", "gru")
        S, orders, cost = mini_rollout(BeerGameParallelEnv(ENVC), actors, enc, mh)
        assert S.mean() > 20, f"S_mean={S.mean():.2f} -- should be ~40 at init, NOT ~0 (the deleted-line bug)"
        assert orders.mean() > 0.5, f"orders {orders.mean():.2f} -- chain should order proactively"
        assert math.isfinite(cost), "cost non-finite"
    run("rollout: S is populated (~40) and ordering is proactive, not S=0 starvation", s_is_populated)

    def both_heads_pipeline():
        for head in ("mlp", "structured"):
            _, enc, actors, mh, _, _ = make_modules(head, "gru")
            S, orders, cost = mini_rollout(BeerGameParallelEnv(ENVC), actors, enc, mh)
            assert S.mean() > 20 and math.isfinite(cost), f"{head} pipeline broken (S={S.mean():.1f})"
    run("rollout runs end-to-end for both heads", both_heads_pipeline)

    def eval_is_deterministic():
        _, enc, actors, mh, _, _ = make_modules("structured", "gru")
        env = BeerGameParallelEnv(ENVC)
        _, _, c1 = mini_rollout(env, actors, enc, mh, deterministic=True)
        _, _, c2 = mini_rollout(env, actors, enc, mh, deterministic=True)
        assert abs(c1 - c2) < 1e-3, f"deterministic eval not reproducible: {c1} vs {c2}"
    run("deterministic eval is reproducible (same seed -> same cost)", eval_is_deterministic)


# ============================================================================= G. CONFIG MATRIX
def test_config_matrix():
    section("G. Config matrix -- every combo constructs and does one update")
    base_axes = [(h, e, c, x)
                 for h in ("mlp", "structured")
                 for e in ("gru", "craft")
                 for c in (True, False)
                 for x in (True, False)]
    for (h, e, comm, ctx) in base_axes:
        def _t(h=h, e=e, comm=comm, ctx=ctx):
            _, _, _, _, _, trn = make_modules(h, e, extra={"use_comm": comm, "use_context": ctx})
            a, c, ee = trn.update(synth_buffers())
            assert all(math.isfinite(x) for x in (a, c, ee))
        run(f"head={h:10s} enc={e:5s} comm={str(comm):5s} ctx={str(ctx):5s}", _t)

    toggles = {
        "belief_sample=True": {"belief_sample": True},
        "risk_eta=0.1": {"risk_eta": 0.1},
        "order_cap_coef=0.05": {"order_cap_coef": 0.05},
        "corr_l2_coef=1e-3": {"corr_l2_coef": 1e-3},
        "critic_uses_belief=False": {"critic_uses_belief": False},
        "s_smooth_coef=0.1": {"s_smooth_coef": 0.1},
        "demand_aux_coef=0.5": {"demand_aux_coef": 0.5},
    }
    for name, extra in toggles.items():
        def _t(extra=extra):
            _, _, _, _, _, trn = make_modules("structured", "gru", extra=extra)
            a, c, ee = trn.update(synth_buffers())
            assert all(math.isfinite(x) for x in (a, c, ee))
        run(f"toggle {name}", _t)


# ============================================================================= main
if __name__ == "__main__":
    print("DRACO v4 component test suite (CPU, no W&B/Hydra)")
    print(f"dims: N={N} obs={OD} z={Z} msg={MSG} gdim={GDIM}")
    test_env()
    test_heads()
    test_encoders()
    test_critic_norm()
    test_trainer()
    test_pipeline()
    test_config_matrix()
    print("\n" + "=" * 60)
    print(f"RESULT: {_PASS} passed, {_FAIL} failed")
    if _FAILED:
        print("FAILED:")
        for n in _FAILED:
            print("   -", n)
    print("=" * 60)
    sys.exit(0 if _FAIL == 0 else 1)