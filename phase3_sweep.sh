#!/usr/bin/env bash
# run_phase3_c2.sh -- PHASE 3: value of communication vs topology, via d_hat BROADCAST.
#
# msg_mode=dhat + msg_dim=1: each agent broadcasts its DETACHED demand belief d_hat as the
# message. No learned channel => msg_penalty_coef / entropy_coef / lr_msg are MOOT (the
# message is a clean extended observation, exactly your Phase-3 design). Topology widens:
#   neighbor (chain) -> skip (range-2) -> full (all-to-all) -> retailer_broadcast (everyone
#   hears the retailer's clean signal undiluted = maximally favorable for the Lee mechanism).
#
# PREREQUISITE -- smoke-test the d_hat broadcast ONCE before launching (it's a new code path):
#   python agents/train_draco_v4.py agent=draco_v4 agent.actor_head=structured \
#     agent.use_comm=true agent.msg_mode=dhat agent.msg_dim=1 agent.comm_topology=full \
#     agent.dr_lambda_lo=4 agent.dr_lambda_hi=24 agent.dr_p_shift=0.0 \
#     total_episodes=80 agent.warm_up_episodes=10 agent.batch_episodes=4 \
#     agent.heldout_every=40 agent.heldout_episodes=2 seed=0 agent.algorithm=smoke_dhat
#   -> confirm it runs, no NaN, and (eval --messages) shows msg<->demand corr ~1.0 (msg IS d_hat).
#
# Usage (SAME seeds as Phase 2 so comm is CRN-paired against the nocomm refs):
#   ./run_phase3_c2.sh 10 11 12 13 14
set -euo pipefail
set -f   # disable globbing so Hydra list overrides like [8,12,16,20] survive bash
source /workspace/venv/bin/activate 2>/dev/null || true   # use the venv from setup_pod.sh
: "${WANDB_API_KEY:?set WANDB_API_KEY first: export WANDB_API_KEY=... (or run: wandb login)}"
cd "${REPO:-/workspace/BeerGame_Project}"   # MUST match your git clone dir; override: REPO=/path ./script.sh  


LOCKED="agent.demand_aux_coef=0.3 agent.z_dim=8 agent.encoder_type=gru"   # from Phase 1
# SELECTION on VALIDATION lambdas (checkpoint/early-stop key off held-out Mean_Cost -> don't
# train on test). Comm value is measured POST-HOC on the test set (see bottom).
VAL="agent.heldout_lambdas=[8,12,16,20]"
BASE="agent=draco_v4 agent.actor_head=structured agent.use_context=true \
      agent.dr_lambda_lo=4 agent.dr_lambda_hi=24 \
      agent.heldout_every=400 agent.heldout_episodes=20 \
      total_episodes=15000 agent.batch_episodes=16 agent.patience=3000 $LOCKED $VAL"
DHAT="agent.msg_mode=dhat agent.msg_dim=1"

for SEED in "$@"; do
  for PS in 0.0 0.4; do
    TAG=$([ "$PS" = "0.0" ] && echo stat || echo nonstat)

    # CRN-paired no-comm reference for THIS demand condition
    python agents/train_draco_v4.py $BASE agent.dr_p_shift=$PS \
      agent.use_comm=false seed=$SEED agent.algorithm=c2_${TAG}_nocomm_s${SEED}

    # d_hat broadcast over widening topologies
    for TOP in neighbor no_neighbor skip full retailer_broadcast; do
      python agents/train_draco_v4.py $BASE agent.dr_p_shift=$PS $DHAT \
        agent.use_comm=true agent.comm_topology=$TOP \
        seed=$SEED agent.algorithm=c2_${TAG}_${TOP}_s${SEED}
    done
  done
done

# ---------------------------------------------------------------------------------------
# ANALYSIS:  python agents/eval_draco_v4.py --ckpt weights_draco/run_.../draco_checkpoint_best.pt --messages
#   value of comm = paired Mean_Cost(topology) - Mean_Cost(nocomm) at matched seeds, per demand
#   condition. EXPECTED: ~0 under stationary (Axsater-Rosling); any positive appears under
#   nonstationary at full/retailer_broadcast (Lee). Both signs publish; the message<->demand
#   correlation (~1.0 by construction) + the upstream-forecast-error delta explain the sign.
#
# STATISTICS: CRN pairing already shrinks variance; use a paired test (Wilcoxon) OR, for a NULL,
# a TOST equivalence test against a +/-delta band. If the nonstationary effect is small but
# nonzero in the 5-seed pilot, scale THAT condition (not the whole grid) to 15-30 seeds.
# ---------------------------------------------------------------------------------------