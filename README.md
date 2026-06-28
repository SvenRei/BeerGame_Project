# DRACO — Regime-Adaptive RL for the Multi-Echelon Beer Game

A reinforcement-learning framework for the 4-echelon serial supply chain ("beer game"). The agent,
**DRACO**, is a cooperative multi-agent actor–critic (MAPPO, centralized training / decentralized
execution) that infers the unknown demand regime from a self-supervised belief and adapts an
interpretable **base-stock** (order-up-to) policy. The repo also ships the classical comparators
(static base-stock, adaptive forecast, Bayes, per-regime oracle), a leakage-controlled
regime-uncertainty benchmark, statistical-inference tooling, demand-distribution robustness, and a
symbolic-distillation pipeline that recovers a closed-form policy from the trained network.

> New to the code? Start with the `READING GUIDE` comment blocks at the top of
> [`agents/draco_v4.py`](agents/draco_v4.py) (the agent) and
> [`agents/train_draco_v4.py`](agents/train_draco_v4.py) (the training loop), then the
> per-step phase map in [`envs/beer_game_env.py`](envs/beer_game_env.py).

---

## Repository layout

```
envs/
  beer_game_env.py        PettingZoo ParallelEnv: 4-echelon beer game, partial obs (4 scalars),
                          configurable lead times, CTDE global state. Cost = h*inv + b*backlog
                          (every stage, or retailer-only via penalty_at_retailer_only).
agents/
  draco_v4.py             The agent: belief encoder (GRU/CRAFT, BAMDP), base-stock actor heads
                          (structured | mlp), DIAL message head, scalar belief-conditioned critic,
                          and the MAPPO/PPO trainer (DRACOTrainerV4).
  train_draco_v4.py       Hydra training entry point (rollout loop, held-out gate, checkpointing).
  eval_draco_v4.py        Evaluation + diagnostics (C1 regime table, OM dashboard, comm/belief
                          probes, --families, --dump-c1 producer for c1_stats).
  heldout_eval.py         Held-out-lambda eval used as the in-training selection gate (C1).
  topologies.py           Communication topologies (neighbor/skip/full/retailer_broadcast/...).
conf/
  config.yaml             Master Hydra config (env block + globals).
  agent/draco_v4.yaml     Agent hyperparameters.
scripts/
  baselines.py            Classical policies + the 4-rung regime benchmark + canonical optimum
                          (validate-canonical) + AR(1)/MMFE comparator. Writes results/baselines_regime_v2.json.
  c1_stats.py             Statistical inference for the C1 result (bootstrap CIs, paired vs Bayes,
                          IQM/performance profiles, Gap<=1 check). `report` CLI + self-test.
  demand_families.py      AR(1) / NegBin / family-randomized demand (subclasses the env; no env edit).
  distill_symbolic.py     C3: PySR + DAgger symbolic distillation of the trained policy.
  play_yourself.py        tkinter GUI to play the env by hand (sanity check the physics).
test/
  test_beer_game_env.py   89 env unit tests (numpy + pettingzoo; no torch).
  test_draco_v4.py        Agent component/integration tests (needs torch).
phase1_sweep.sh           Study 1: architecture HP sweep (no comm).      } bash; run on a GPU host.
phase2_sweep.sh           Study 2: the C1 spine (locked config, seeds).  } Each reads the one
phase3_sweep.sh           Study 3: comm topology sweep (d_hat).          } baselines_regime_v2.json.
phase3b_ar1_comm_sweep.sh Study 3b: comm value vs AR(1) autocorrelation.
run_phase1_cluster.py     Parallel phase-1 launcher (pod-specific; see "Sweeps" caveat below).
setup_local.sh            One-shot local venv + deps + smoke (Git Bash / macOS / Linux).
setup_pod.sh              Cloud-pod bootstrap (RunPod; pod-specific paths — see caveat).
smoke_test.sh             Fast end-to-end smoke of the whole project.
requirements.txt          Full deps (CUDA 12.1 torch + pysr/Julia).
```

Generated at runtime (git-ignored): `weights_draco/` (checkpoints), `results/` (refs + dumps),
`outputs/` (Hydra run dirs), `wandb/`, `run_logs/`.

---

## Architecture (one screen)

- **Environment** — serial chain retailer → wholesaler → distributor → manufacturer. Each agent
  observes only 4 local scalars `[inventory, backlog, on_order, last_demand]`; a centralized critic
  sees the full pipeline during training only (CTDE). Per-period cost is `h·inventory + b·backlog`
  at every echelon by default (the behavioral beer-game team cost), or retailer-only with
  `env.penalty_at_retailer_only=true` (the canonical Clark–Scarf cost, which has a provable optimum).
- **Belief (BAMDP)** — a causal, action-free encoder (GRU or CRAFT transformer) maps the observation
  history to a latent `z`, trained by a self-supervised demand-reconstruction ELBO so `z` encodes the
  demand regime. `z` is detached for the policy/critic (the encoder is a clean forecaster).
- **Actor head** — emits an order-up-to level `S`; the env order is `clip(S − IP)`. `structured`
  bakes in `S = lead·d̂ + safety + bounded_correction` (interpretable, extrapolates with demand);
  `mlp` is the unconstrained control and the substrate for symbolic distillation.
- **Communication (optional)** — agents broadcast a message (`dhat` = their detached demand belief,
  or a `learned` DIAL vector) routed along a topology with a one-step delay.

---

## Install

**Full (GPU, for training/eval/distill):**
```bash
python -m pip install -r requirements.txt   # CUDA 12.1 torch + pysr (Julia installs on first import)
```
`requirements.txt` pins the CUDA-12.1 torch wheels; on a CPU-only box install
`torch --index-url https://download.pytorch.org/whl/cpu` instead. `pysr` is only needed for the C3
distillation (its Julia backend installs lazily on first use).

**Lightweight (env + scripts only, no torch):**
```bash
python -m pip install numpy scipy pettingzoo gymnasium pytest
```

---

## Quickstart: smoke test

Verifies every component runs (tiny settings; not a performance test). Auto-skips the torch stages
if torch is absent.

```bash
bash setup_local.sh          # creates .venv, installs lightweight deps, runs the smoke suite
bash setup_local.sh --full   # also installs torch (CPU) + wandb/hydra → agent/training/eval smoke
# already set up? re-run anytime:
PY=.venv/Scripts/python.exe bash smoke_test.sh        # Windows venv path; POSIX: .venv/bin/python
```

Windows without Git Bash — the same in `cmd`:
```bat
python -m venv .venv && .venv\Scripts\activate && python -m pip install numpy scipy pettingzoo gymnasium pytest
python test\test_beer_game_env.py & python scripts\c1_stats.py & python scripts\demand_families.py & python scripts\distill_symbolic.py
```

---

## The research pipeline

The benchmark is generated **once** and read by every training run and every eval — do not
regenerate it per run. (`set WANDB_MODE=disabled` / `export WANDB_MODE=disabled` to skip W&B login.)

```bash
# 1) BENCHMARK (once): the 4-rung ladder -> results/baselines_regime_v2.json
python scripts/baselines.py regime --lambdas 6 10 14 18 22 --select-episodes 80 --eval-episodes 200

# 2) TRAIN (the trainer auto-loads the refs above; logs Gap vs BAR/Oracle/Bayes)
python agents/train_draco_v4.py agent=draco_v4 agent.actor_head=structured agent.use_comm=false \
       agent.demand_aux_coef=0.3 agent.z_dim=8 agent.encoder_type=gru \
       total_episodes=15000 seed=10 agent.algorithm=c1_s10

# 3) EVAL a checkpoint (standard benchmark + C1 table + held-out families + bullwhip comparison)
python agents/eval_draco_v4.py --ckpt weights_draco/run_dracov4_<id>/draco_checkpoint_best.pt \
       --episodes 100 --regime-episodes 20 --families --bullwhip

# 4) STATISTICS over seeds: dump per-seed costs, then aggregate
python agents/eval_draco_v4.py --ckpt <ckpt> --dump-c1 results/draco_c1 --dump-c1-episodes 200
python scripts/c1_stats.py report --draco-dir results/draco_c1 --refs results/baselines_regime_v2.json

# C3) SYMBOLIC DISTILLATION (distill the trained policy into a closed-form rule)
python scripts/distill_symbolic.py --ckpt <ckpt> --backend pysr --dagger-rounds 4
```

For the statistical headline, train ≥5 (ideally ≥10) seeds (loop `seed=10..14`), `--dump-c1` each,
then run `c1_stats report` over all of them.

### Sweeps
`phase{1,2,3,3b}_sweep.sh` are GPU-host scripts that run the multi-config / multi-seed studies; each
arm reads the same `results/baselines_regime_v2.json`. They contain `/workspace`-style paths and a
`WANDB_API_KEY` requirement — override `REPO=` / `VENV=` env vars or edit the header for your host.
`run_phase1_cluster.py` is a pod-specific parallel launcher and is **not** parameter-aligned with the
bash scripts (see "Known limitations").

---

## Configuration

Hydra composes `conf/config.yaml` + `conf/agent/draco_v4.yaml`; override anything on the CLI
(`key=value`). Forward slashes in paths work on Windows too.

| Override | Values | Meaning |
|---|---|---|
| `agent.actor_head` | `structured` \| `mlp` | base-stock-grounded head vs. unconstrained (SR substrate) |
| `agent.use_comm` | `false` \| `true` | enable inter-agent messages |
| `agent.msg_mode` | `dhat` \| `learned` | broadcast the demand belief vs. a learned DIAL vector |
| `agent.comm_topology` | `neighbor` \| `skip` \| `full` \| `retailer_broadcast` \| `no_neighbor` | message routing |
| `agent.dr_mode` | `poisson` \| `family` | randomize Poisson rate vs. demand family (Poisson/NegBin/AR1) |
| `agent.dr_families` | e.g. `[poisson,negbin]` | training families when `dr_mode=family` |
| `agent.encoder_type` | `gru` \| `craft` | belief encoder |
| `agent.s_smooth_coef`, `agent.corr_l2_coef` | float | bullwhip control (smooth S / shrink correction) |
| `agent.risk_eta` | float (0 = off) | optional CVaR tail-risk reweighting (see caveats) |
| `env.penalty_at_retailer_only` | `false` \| `true` | beer-game cost vs. canonical Clark–Scarf cost |
| `env.demand_type` | `poisson` \| `step` \| `zero` \| `black_swan` \| `extreme_chaos` | demand process |
| `total_episodes`, `seed` | int | run length / RNG seed (top-level, not under `agent.`) |

---

## Testing

```bash
python test/test_beer_game_env.py      # 89 env unit tests (no torch); or: pytest test/test_beer_game_env.py
python test/test_draco_v4.py           # agent component/integration tests (needs torch)
bash smoke_test.sh                     # everything, tiny settings
```

---

## Known limitations & scientific caveats

Read these before reporting numbers — they shape what the results mean.

- **Cost model vs. optimality.** The default cost penalizes backlog at *every* echelon, so the
  reported "oracle" is a numerically-optimized static base-stock, **not** a provable optimum. For a
  provable ceiling, use `env.penalty_at_retailer_only=true` (canonical Clark–Scarf cost) and validate
  with `python scripts/baselines.py validate-canonical`.
- **Adaptation induces bullwhip.** Under the penalty-at-every-stage cost, a static base-stock is
  ~bullwhip-free, while *any* adaptive policy (Bayes, EWMA, and a naïve DRACO) injects order variance
  that is costly upstream. Empirically the static base-stock beats the adaptive comparators; the
  per-regime oracle is a *regime-matched, bullwhip-free* policy. The headline comparators are the
  **static base-stock and the oracle**; treat Bayes/EWMA as a "naïve-adaptation floor." Use
  `agent.s_smooth_coef` to push DRACO toward adapting without bullwhipping.
- **Statistics need seeds.** With 5 seeds the nonparametric p-value is floored near 0.06; report ≥10
  seeds with bootstrap CIs / IQM (`scripts/c1_stats.py`).
- **C1 tests interpolation.** Test λ ⊂ the training support, so C1 measures regime *inference*, not
  extrapolation; out-of-distribution is tested via the demand families and `black_swan`/`extreme_chaos`.
- **Risk path is dormant and heuristic.** `agent.risk_eta` defaults to 0 and the implementation is an
  advantage reweight, not a principled CVaR method — do not claim CVaR-optimality.
- **Pod/cluster scripts are host-specific.** `setup_pod.sh` and `run_phase1_cluster.py` hardcode
  cloud paths/hardware and the latter's episode/patience settings differ from the bash sweeps; prefer
  the bash sweeps and override paths via env vars.
