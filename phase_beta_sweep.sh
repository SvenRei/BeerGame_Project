#!/usr/bin/env bash
# phase_beta_sweep.sh -- ABLATION on the SRDQN credit-assignment weight beta (review #4).
#
# Each agent's shaped reward is  -(own_cost + beta * others_cost).
#   beta = 1.0  -> reward = -(total team cost) for every agent = EXACTLY the evaluated objective
#                  (team-optimal; objective-exact).
#   beta < 1.0  -> down-weights others' cost: lower-variance signal, but NOT the reported objective.
# This sweep reports how sensitive the C1 result is to beta in {0.0, 0.5, 1.0}, and lets you state
# that the headline either uses the objective-exact beta=1.0 or that performance is beta-insensitive.
#
# Usage (same report seeds as Phase 2, for CRN pairing):  ./phase_beta_sweep.sh 10 11 12 13 14
set -euo pipefail
set -f   # keep Hydra list overrides like [8,12,16,20] intact through bash
source "${VENV:-/workspace/venv}/bin/activate" 2>/dev/null || true
: "${WANDB_API_KEY:?set WANDB_API_KEY first, or: export WANDB_MODE=disabled}"
cd "${REPO:-/workspace/BeerGame_Project}"

LOCKED="agent.demand_aux_coef=0.3 agent.z_dim=8 agent.encoder_type=gru"   # Phase-1 winner
VAL="agent.heldout_lambdas=[8,12,16,20]"                                  # select on VAL (anti-leakage)
BASE="agent=draco_v4 agent.actor_head=structured agent.use_comm=false agent.use_context=true \
      agent.dr_lambda_lo=4 agent.dr_lambda_hi=24 agent.dr_p_shift=0.0 \
      agent.heldout_every=400 agent.heldout_episodes=20 \
      total_episodes=15000 agent.batch_episodes=8 agent.patience=3000 $LOCKED $VAL"

for SEED in "$@"; do
  for BETA in 0.0 0.5 1.0; do
    python agents/train_draco_v4.py $BASE agent.srdqn_beta=$BETA \
      seed=$SEED agent.algorithm=beta${BETA}_s${SEED}
  done
done

# ---------------------------------------------------------------------------------------
# ANALYSIS (post-hoc on the TEST lambdas): per beta, dump per-seed costs then aggregate:
#   for each beta's checkpoints:
#     python agents/eval_draco_v4.py --ckpt weights_draco/run_.../draco_checkpoint_best.pt \
#       --dump-c1 results/beta_${BETA} --dump-c1-episodes 200
#   python scripts/c1_stats.py report --draco-dir results/beta_${BETA} --refs results/baselines_regime_v2.json
# Report Gap_Recovered mean [95% CI] vs beta. Decision rule: if beta=0.5 is NOT significantly better
# than beta=1.0, report the OBJECTIVE-EXACT beta=1.0 as the headline. Also check (eval --full) whether
# beta changes the bullwhip (order-variance amplification) -- the S2 mechanism.
# ---------------------------------------------------------------------------------------
