#!/usr/bin/env bash
set -euo pipefail
cd /workspace/BeerGame                       # repo root on the pod
export WANDB_API_KEY=...                      # your key
# pip install -r requirements.txt            # if not baked into the image

# >>> the locked C1 config (append any L2 overrides you kept, e.g. agent.demand_aux_coef=0.3) <<<
COMMON="agent=draco_v4 \
        agent.dr_lambda_lo=4 agent.dr_lambda_hi=24 agent.dr_p_shift=0.0 \
        agent.heldout_every=400 agent.heldout_episodes=20 \
        total_episodes=8000 agent.batch_episodes=16 agent.patience=3000"

for SEED in 0 1 2 3 4; do
  # (1) C1 main: structured, no comm
  python agents/train_draco_v4.py $COMMON \
    agent.actor_head=structured agent.use_comm=false \
    seed=$SEED agent.algorithm=c1_struct_nocomm_s${SEED}

  # (2) H4 comm-null ablation: structured, neighbor comm
  python agents/train_draco_v4.py $COMMON \
    agent.actor_head=structured agent.use_comm=true \
    seed=$SEED agent.algorithm=c1_struct_neighbor_s${SEED}

  # (3) negative control: MLP head, no comm
  python agents/train_draco_v4.py $COMMON \
    agent.actor_head=mlp agent.use_comm=false \
    seed=$SEED agent.algorithm=c1_mlp_nocomm_s${SEED}
done