#!/usr/bin/env bash
# phase3b_ar1_comm_sweep.sh -- STUDY 3 (deepened): value of communication vs demand AUTOCORRELATION.
#
# The serial-chain comm null (Axsater-Rosling 1993: installation = echelon, so local info already
# suffices at the optimum) is the STATIONARY prediction. Lee, So & Tang (2000) predict comm value
# GROWS with demand autocorrelation rho. This sweep measures exactly that curve: comm value vs rho.
#
# It trains DRACO directly on AR(1) demand at each rho, via the family-DR machinery already in the
# trainer (no new code): dr_mode=family, dr_families=[ar1], with the AR(1) rho/mu PINNED. Arms per rho:
#   - no-comm reference
#   - dhat broadcast over topologies {neighbor, full, retailer_broadcast}   (the interpretable channel)
#   - a LEARNED VECTOR message (msg_mode=learned, msg_dim=3)                 (capacity upper bound; Q4)
# The learned-vector arm answers "does a higher-capacity, opaque message buy more than sharing d_hat?"
# If it does not, the null is stronger; if it does, the message-content probes (eval --full) say why.
#
# WHY pin rho per run: it isolates the rho effect cleanly (one comm/no-comm pair per rho). The cheaper
# alternative is to train ONCE on rho ~ U[0,0.9] and evaluate per-rho with eval --ar1-rhos; do that if
# compute is tight (1 pair of checkpoints instead of 4).
#
# Usage (same >=10 seeds as Phase 2 for CRN pairing):  ./phase3b_ar1_comm_sweep.sh 10 11 12 13 14 15 16 17 18 19
set -euo pipefail
set -f   # keep Hydra list overrides like [ar1] / [6,10,14,18,22] intact through bash
source "${VENV:-/workspace/venv}/bin/activate" 2>/dev/null || true
: "${WANDB_API_KEY:?set WANDB_API_KEY first: export WANDB_API_KEY=... (or run: wandb login)}"
cd "${REPO:-/workspace/BeerGame_Project}"

LOCKED="agent.demand_aux_coef=0.3 agent.z_dim=8 agent.encoder_type=gru"   # the Phase-1 winner
# Train ON AR(1) and gate (checkpoint/early-stop) on held-out AR(1) at VALIDATION rho {0.15,0.45,0.75}
# (disjoint from the test rho below) so the selection target matches the AR(1) study objective.
# The comm VALUE is measured POST-HOC on the matched test rho (see ANALYSIS at the bottom).
BASE="agent=draco_v4 agent.actor_head=structured agent.use_context=true \
      agent.heldout_mode=ar1 agent.heldout_ar1_rhos=[0.15,0.45,0.75] \
      agent.heldout_every=400 agent.heldout_episodes=20 \
      total_episodes=15000 agent.batch_episodes=8 agent.patience=3000 $LOCKED"
DHAT="agent.msg_mode=dhat agent.msg_dim=1"
VEC="agent.msg_mode=learned agent.msg_dim=3 agent.lr_msg=3.0e-4 agent.msg_penalty_coef=0.0"

for SEED in "$@"; do
  for RHO in 0.0 0.3 0.6 0.9; do
    AR1="agent.dr_mode=family agent.dr_families=[ar1] \
         agent.ar1_mu_lo=12 agent.ar1_mu_hi=12 agent.ar1_rho_lo=$RHO agent.ar1_rho_hi=$RHO agent.ar1_sigma=3"
    TAG=rho${RHO}

    # no-comm reference for THIS rho
    python agents/train_draco_v4.py $BASE $AR1 \
      agent.use_comm=false seed=$SEED agent.algorithm=ar1_${TAG}_nocomm_s${SEED}

    # d_hat broadcast over widening topologies (the interpretable Lee channel)
    for TOP in neighbor full retailer_broadcast; do
      python agents/train_draco_v4.py $BASE $AR1 $DHAT \
        agent.use_comm=true agent.comm_topology=$TOP seed=$SEED agent.algorithm=ar1_${TAG}_${TOP}_s${SEED}
    done

    # learned VECTOR message (capacity upper bound; the Q4 dhat-vs-vector comparison)
    python agents/train_draco_v4.py $BASE $AR1 $VEC \
      agent.use_comm=true agent.comm_topology=full seed=$SEED agent.algorithm=ar1_${TAG}_vec_s${SEED}
  done
done

# ---------------------------------------------------------------------------------------
# ANALYSIS (post-hoc, per rho, CRN-paired): for each checkpoint, score it on AR(1) at its OWN rho:
#   python agents/eval_draco_v4.py --ckpt weights_draco/run_.../draco_checkpoint_best.pt \
#     --families --ar1-rhos $RHO --family-episodes 200
# comm value(rho, topology) = cost(nocomm) - cost(topology) at matched rho/seed.
#   EXPECT ~0 at rho=0 (Axsater-Rosling); rising with rho (Lee, So & Tang 2000). Report comm value
#   as a function of rho with the pre-registered TOST band (c1_stats.paired) -> an honest null/effect.
#   Use eval --full to get the message-content / see-through-bullwhip probes that EXPLAIN the sign,
#   and to compare the dhat channel vs the learned-vector channel (does extra capacity buy more?).
# ---------------------------------------------------------------------------------------
