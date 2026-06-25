#!/usr/bin/env bash
# run_phase2_c1.sh -- PHASE 2: the C1 headline, with the LOCKED config from Phase 1.
#
# Two anti-leakage rules, both load-bearing for the publication:
#   1. TEST lambdas {6,10,14,18,22} (Phase 1 selected on the disjoint validation set).
#   2. FRESH seeds {10,11,12,13,14}, disjoint from the Phase-1 selection seeds {0,1,2}.
# This makes Phase 2 an honest held-out estimate, not a re-report of the tuning objective.
#
# Usage (5 fresh report seeds; split across pods if you like):
#   ./run_phase2_c1.sh 10 11 12 13 14
set -euo pipefail
set -f   # disable globbing so Hydra list overrides like [8,12,16,20] survive bash
source /workspace/venv/bin/activate 2>/dev/null || true   # use the venv from setup_pod.sh
: "${WANDB_API_KEY:?set WANDB_API_KEY first: export WANDB_API_KEY=... (or run: wandb login)}"
cd "${REPO:-/workspace/BeerGame_Project}"   # MUST match your git clone dir; override: REPO=/path ./script.sh  


# >>>>> PASTE THE LOCKED HPs FROM PHASE 1 HERE (these are the validated defaults as a fallback) <<<<<
LOCKED="agent.demand_aux_coef=0.3 agent.z_dim=8 agent.encoder_type=gru"

# SELECTION on VALIDATION lambdas {8,12,16,20}: checkpoint + early-stop now key off the
# held-out Mean_Cost, so training must NOT see the test lambdas, or the saved checkpoint is
# selected on the test set. The headline TEST number is computed POST-HOC (see bottom).
VAL="agent.heldout_lambdas=[8,12,16,20]"
BASE="agent=draco_v4 agent.use_comm=false \
      agent.actor_head=structured agent.use_context=true \
      agent.dr_lambda_lo=4 agent.dr_lambda_hi=24 agent.dr_p_shift=0.0 \
      agent.heldout_every=400 agent.heldout_episodes=20 \
      total_episodes=15000 agent.batch_episodes=16 agent.patience=3000 $LOCKED $VAL"
# total_episodes=10000: the C1 spine was still climbing at 6000 -> give it room to clear the bar.

for SEED in "$@"; do
  # C1 SPINE: the headline regime-inference result (mean +/- std Gap_Recovered over seeds)
  python agents/train_draco_v4.py $BASE seed=$SEED agent.algorithm=c1_spine_s${SEED}
done

# ---------------------------------------------------------------------------------------
# HEADLINE TEST NUMBER (computed ONCE, post-hoc, on the val-selected checkpoints -> no leak):
#   for each c1_spine_s* checkpoint:
#     python agents/eval_draco_v4.py --ckpt weights_draco/run_.../draco_checkpoint_best.pt \
#       --bar 4726 --ceiling 2202 --regime-episodes 20
#   report mean +/- std of the regime-uncertainty Gap_Recovered over the 5 seeds. The
#   eval defaults already use the TEST lambdas {6,10,14,18,22}.

# ---------------------------------------------------------------------------------------
# ARCHITECTURE ABLATION (Study 4 / H3) -- run your run_ablations_sweep.sh, but FIRST add the
# locked HPs to its BASE_OPTS and use these SAME fresh seeds and TEST refs, so the 2x2
# (structured/mlp x context/nocontext) is measured against the identical bar. The expected
# result you already saw locally: only ablate_full crosses the bar; the other three don't.
# ---------------------------------------------------------------------------------------