#!/usr/bin/env bash
# run_phase1_hpsweep.sh -- PHASE 1: hyperparameter sweep (non-comm, structured head).
#
# Sweeps the THREE architecture-defining HPs that govern the BAMDP + structured-head
# mechanism:  demand_aux_coef (grounding) x z_dim (belief capacity) x encoder_type.
# PPO HPs are fixed at standard values (not the contribution; a reviewer won't demand them).
#
# CRITICAL (no leakage): selection is on a VALIDATION lambda set {8,12,16,20}, DISJOINT from
# the C1 test set {6,10,14,18,22} used in Phase 2. Rank configs by Eval_lambda/Mean_Cost
# (lower = better) -- this ranking is ref-independent, so the heldout_*_ref defaults are
# irrelevant here (ignore Gap_Recovered in Phase 1; it's miscomputed against test refs).
#
# 18 configs x the seeds you pass. Parallelize BY SEED across pods:
#   pod A: ./run_phase1_hpsweep.sh 0     pod B: ... 1     pod C: ... 2
set -euo pipefail
set -f   # disable globbing so Hydra list overrides like [8,12,16,20] survive bash
source /workspace/venv/bin/activate 2>/dev/null || true   # use the venv from setup_pod.sh
: "${WANDB_API_KEY:?set WANDB_API_KEY first: export WANDB_API_KEY=... (or run: wandb login)}"
cd "${REPO:-/workspace/BeerGame_Project}"   # MUST match your git clone dir; override: REPO=/path ./script.sh                      # <-- make this consistent across ALL your scripts


VAL="agent.heldout_lambdas=[8,12,16,20]"       # validation split (NOT the C1 test lambdas)
BASE="agent=draco_v4 agent.use_comm=false \
      agent.actor_head=structured agent.use_context=true \
      agent.dr_lambda_lo=4 agent.dr_lambda_hi=24 agent.dr_p_shift=0.0 \
      agent.heldout_every=400 agent.heldout_episodes=20 \
      total_episodes=15000 agent.batch_episodes=16 agent.patience=3000 $VAL"

for SEED in "$@"; do
  for AUX in 0.1 0.3 0.5; do                   # grounding strength (0.3 validated; brackets it)
    for Z in 4 8 16; do                        # belief capacity (8 default; brackets it)
      for ENC in gru craft; do                 # encoder architecture (drop 'craft' to halve cost)
        python agents/train_draco_v4.py $BASE \
          agent.demand_aux_coef=$AUX agent.z_dim=$Z agent.encoder_type=$ENC \
          seed=$SEED agent.algorithm=p1_aux${AUX}_z${Z}_${ENC}_s${SEED}
      done
    done
  done
done

# SELECTION: for each (aux,z,enc), average Eval_lambda/Mean_Cost over seeds; pick the MINIMUM.
# That (aux,z,enc) is the LOCKED config -> paste it into run_phase2_c1.sh and run_phase3_c2.sh.
# Report the full grid as a sensitivity table (shows the result isn't a knife-edge).