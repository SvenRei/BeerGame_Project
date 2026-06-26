# BeerGame

Paper: Regime-Adaptive Decentralized Ordering in Serial Supply Chains: What Communication Can and Cannot Buy, and an Interpretable Distillation

This repository contains the official implementation of DRACO (Decentralized Regime-Adaptive Control). It provides a full reinforcement learning and evaluation framework for multi-agent serial supply chains facing regime uncertainty (non-stationary demand distributions).

## Core Methodology

This project is built around two foundational theorems in Operations Research:

The repository evaluates two core algorithmic families across communicating and non-communicating variants:

    Clark & Scarf (1960): Echelon base-stock is optimal for serial systems with linear costs.

        

DRACO does not attempt to "beat" the Clark-Scarf optimum. Instead, it tackles the reality of deployment: classical base-stock policies must commit to a single inventory level across an unknown demand distribution, making them badly suboptimal under regime uncertainty. DRACO leverages context-based meta-RL (BAMDP) to infer the current demand regime and adapt its order-up-to levels dynamically, crushing the best regime-agnostic fixed base-stock policy while isolating the true marginal value of inter-agent communication.

## File Structure

```
├── agents/
│   └── rl/
│       └── comm_utils.py          # draco_v4.py
├── configs/                       # Hydra environment configuration mappings
├── envs/
│   └── beer_game_env.py       # Custom PettingZoo/Gymnasium multi-agent Beer Game environment
├── sweep_draco.yaml          # W&B Bayesian Sweep configuration for Comm-MAPPO
├── sweep_comm.yaml           # W&B Bayesian Sweep configuration for Comm-QMIX
├── README.md              # W&B Bayesian Sweep configuration for baseline QMIX
└── requirements.txt               # Complete Python system and training dependencies
```

## Theoretical Backbone

| Result | Reference | Role in the Paper |
| :--- | :--- | :--- |
| **Echelon base-stock optimal, serial** | Clark & Scarf 1960 | The optimum / ceiling; why "beat base-stock" is impossible. |
| **Installation = echelon (local sufficient)** | Axsäter & Rosling 1993 | Why comm is null at the optimum (H4); turns a null into a theorem. |
| **Bullwhip from demand-signal-processing; value of info sharing** | Lee, Padmanabhan & Whang 1997 | The only mechanism for C2 > 0; predicts skip-level helps under non-stationarity. |
| **Base-stock optimal under forecast evolution (MMFE)** | Heath & Jackson 1994 | Justifies belief → order-up-to-level under non-stationary demand. |
| **Generalization of base-stock-regularized policies** | VC-theory, arXiv:2404.11509 | Why a structured/symbolic policy should extrapolate (H6). |
| **BAMDP / context-based meta-RL** | VariBAD, PEARL | The formal frame for the belief encoder + regime inference (C1). |
| **Symbolic policy distillation; distribution-shift caution** | SPID 2025; PySR (Cranmer 2023) | Method and honest fidelity reporting (C3). |

## Planned study

### Study 1 — C1: regime inference (the spine)

    Train: DRACO (no-comm) on the demand-randomization curriculum (λ ~ U[lo,hi] per episode + occasional within-episode shift).
    Eval (the key table): a held-out set of stationary λ ∈ {6, 10, 14, 18, 22} (per-episode constant, unknown to the agent), scored against (a) best single fixed S, (b) per-λ oracle, (c) Sterman.
    Primary metric: mean cost per regime; and the headline scalar fraction of the (fixed − oracle) gap recovered.
    A lso: structure-recovery plot for H3 (learned S vs lead·d̂ + safety); in-support match for H1 (DRACO vs oracle on the training λ band).
    Ablation (folds in point 1's comm-null): within Study 1, add neighbor-comm as one arm and show Δcost ≈ 0 vs no-comm (H4).


### Study 2 — C2: value-of-communication topology sweep

    Manipulation: topology ∈ {no-comm, neighbor, skip-level, full} × demand stochasticity ∈ {stationary in-support, non-stationary (within-episode shifts), shock (black_swan/extreme_chaos)}.
    Message: fixed content = sender's d̂ (Section 5).
    Primary metric: paired Δcost vs no-comm (Wilcoxon over common seeds), per cell; reported as a topology × stochasticity surface.
    Prediction: flat-≈0 under stationary (theorem); if anything is positive, it appears skip-level/full under non-stationarity (Lee et al.). Either outcome is the result — a flat surface is empirical confirmation of installation=echelon optimality in a learned setting, which is itself novel and citable.
    Diagnostic (explains the sign): message-informativeness probe — does the shared d̂ reduce upstream forecast error vs forecasting from local orders? If comm helps, this must move; if it doesn't, that is the explanation for the null.


### Study 3 — C3: symbolic distillation

    Winner: the best Study-1 DRACO (the regime-adaptive policy).
    Distill: PySR over (belief/observed features) → effective order-up-to level S, per stage. Expect to recover something close to lead·d̂ + safety with small corrections.
    Test (H6): evaluate neural vs symbolic vs DAgger-refined symbolic on in-support and out-of-support regimes.
    Report: OOD cost (does symbolic extrapolate ≥ neural?) and fidelity both on neural trajectories and under the symbolic policy's own rollouts (the gap quantifies distribution-shift inflation — the SPID caution).

## DRACO v4 Architecture

DRACO v4 implements a **Centralized Training, Decentralized Execution (CTDE)** architecture. During deployment, agents rely strictly on localized physical observations. During training, a centralized critic leverages the omniscient global state and localized demand beliefs to stabilize the PPO gradient.

```text
================================================================================
                      DRACO v4 Multi-Agent Architecture
================================================================================

      [ ENVIRONMENT: Supply Chain / Beer Game ]
               |                    |
               | (Partial View)     | (Omniscient View - TRAINING ONLY)
               v                    v
+-----------------------------+  +---------------------------------------------+
| LOCAL OBSERVATION (obs)     |  | GLOBAL STATE (state)                        |
| Node-specific physical data:|  | Concatenated physical data of ALL nodes     |
| Inventory, Backlog, Orders  |  | (Retailer + Wholesaler + Dist + Manu)       |
+-----------------------------+  +---------------------------------------------+
               |                                       |
               |                                       |
               v                                       |
  [ OPTIONAL MSG CHANNEL ]                             |
  . - - - - - - - - - - - .                            |
  :  Incoming Msg (msg_in):  if use_comm=True          |
  :  (Zeros)              :  if use_comm=False         |
  . - - - - - - - - - - - .                            |
               |                                       |
               v                                       v
+-----------------------------+          +-----------------------------+
| MODULE B: Encoder           |          | MODULE D: Value Critic      |
| (Action-Free, Causal)       |          | (Centralized MAPPO Critic)  |
| GRU or CRAFT Transformer    |          | Value = Critic(state, z)    |
+-----------------------------+          +-----------------------------+
               |                                       ^
               v                                       |
       [ DEMAND BELIEF (z) ] --------------------------+ (detached)
               |
               +----------------------------------+
               |                                  |
               v                                  v
+-----------------------------+  . - - - - - - - - - - - - - - - - - .
| MODULE A: Structured Actor  |  : [ OPTIONAL ]                      :
| 1. d_hat = Linear(z)        |  : MODULE C: Message Head            :
| 2. base = lead*d_hat + safe |  : Out_Msg = DIAL(obs, z) OR d_hat   :
| 3. corr = tanh(MLP(obs,z,m))|  :                                   :
| S = softplus(base + corr)   |  :-----> Broadcast via Topology ---->:
+-----------------------------+  . - - - - - - - - - - - - - - - - - .
               |
               v
     [ TARGET BASE-STOCK (S) ]
               |
               v
  Action = max(0, S - Inventory)