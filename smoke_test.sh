#!/usr/bin/env bash
# smoke_test.sh -- fast end-to-end smoke tests for the WHOLE DRACO project.
#
# Runs every component at TINY settings to catch breakage fast after code changes. This is a
# "does it run / wire up" check, NOT a correctness or performance test (episode counts are far too
# small for meaningful numbers). Non-torch stages run anywhere; the agent/training/eval/distill
# stages run only if torch+wandb+hydra are importable (auto-skipped otherwise).
#
# Usage:
#   bash smoke_test.sh                 # full smoke (auto-skips torch stages if torch is missing)
#   PY=/path/to/python bash smoke_test.sh   # choose the interpreter (default: python)
#
# Torch stages take a few minutes on CPU (7 tiny trainings + eval + distill); seconds on a GPU pod.
# Smoke checkpoints land in weights_draco/run_dracov4_*  and results/smoke_c1/ -- delete when done.
set -uo pipefail                         # NOT -e: we want to run every stage and tally pass/fail
PY="${PY:-python}"
export WANDB_MODE="${WANDB_MODE:-disabled}"   # no wandb login / no network during smoke
export PYTHONWARNINGS="${PYTHONWARNINGS:-ignore}"
cd "$(dirname "$0")"

pass=0; fail=0; skip=0
run() {                                  # run "<name>" <cmd...>
  local name="$1"; shift
  echo "------------------------------------------------------------"
  echo ">> $name"
  if "$@"; then echo "   PASS: $name"; pass=$((pass+1))
  else          echo "   FAIL: $name (exit $?)"; fail=$((fail+1)); fi
}
have() { "$PY" -c "import $1" >/dev/null 2>&1; }

echo "============================================================"
echo "DRACO SMOKE TESTS   (interpreter=$PY, WANDB_MODE=$WANDB_MODE)"
echo "============================================================"

# ---------------- 1. NON-TORCH self-tests (run anywhere) ----------------
run "env unit tests (test_beer_game_env, 89 tests)" "$PY" test/test_beer_game_env.py
run "c1_stats self-test"             "$PY" scripts/c1_stats.py
run "demand_families self-test"      "$PY" scripts/demand_families.py
run "distill_symbolic self-test"     "$PY" scripts/distill_symbolic.py
run "env smoke (default + canonical cost)" "$PY" -c "
from envs.beer_game_env import BeerGameParallelEnv
for flag in (False, True):
    e=BeerGameParallelEnv({'demand_type':'poisson','penalty_at_retailer_only':flag}); o,_=e.reset(seed=0); d=False
    while e.agents and not d:
        o,_,_,t,i=e.step({a:[0.3] for a in e.agents}); d=any(t.values())
    assert i['retailer']['supply_chain_cost']>0
print('env ok (default + canonical)')"
run "baselines AR(1)/forecast helpers" "$PY" -c "
from scripts.baselines import ar1_cumulative_forecast
m,s=ar1_cumulative_forecast(18,12,0.7,3,5); assert m>0 and s>0; print('baselines helpers ok', round(m,1), round(s,1))"

# ---------------- 2. TORCH stages (agent / training / eval / distill) ----------------
if have torch && have wandb && have hydra; then
  run "agent component tests (test_draco_v4)" "$PY" test/test_draco_v4.py

  # tiny trainings -- one per NEW/changed code path. Each writes a checkpoint under weights_draco/.
  C="agent=draco_v4 total_episodes=16 agent.warm_up_episodes=2 agent.batch_episodes=4 \
     agent.heldout_every=6 agent.heldout_episodes=1 agent.eval_every=6 agent.eval_episodes=1 seed=0"
  run "train: structured, no-comm (the main path)" "$PY" agents/train_draco_v4.py $C agent.actor_head=structured agent.use_comm=false agent.algorithm=smoke_struct
  run "train: mlp head (SR substrate)"             "$PY" agents/train_draco_v4.py $C agent.actor_head=mlp        agent.use_comm=false agent.algorithm=smoke_mlp
  run "train: comm dhat broadcast"                 "$PY" agents/train_draco_v4.py $C agent.actor_head=structured agent.use_comm=true agent.msg_mode=dhat agent.msg_dim=1 agent.comm_topology=full agent.algorithm=smoke_dhat
  run "train: comm learned vector (msg_dim=3)"     "$PY" agents/train_draco_v4.py $C agent.actor_head=structured agent.use_comm=true agent.msg_mode=learned agent.msg_dim=3 agent.algorithm=smoke_vec
  run "train: family DR (poisson+negbin)"          "$PY" agents/train_draco_v4.py $C agent.actor_head=structured agent.dr_mode=family 'agent.dr_families=[poisson,negbin]' agent.algorithm=smoke_fam
  run "train: AR(1) family + dhat comm"            "$PY" agents/train_draco_v4.py $C agent.actor_head=structured agent.dr_mode=family 'agent.dr_families=[ar1]' agent.ar1_rho_lo=0.6 agent.ar1_rho_hi=0.6 agent.use_comm=true agent.msg_mode=dhat agent.msg_dim=1 agent.algorithm=smoke_ar1
  run "train: risk on (CVaR path)"                 "$PY" agents/train_draco_v4.py $C agent.actor_head=structured agent.risk_eta=0.2 agent.algorithm=smoke_risk
  run "train: canonical cost variant"             "$PY" agents/train_draco_v4.py $C agent.actor_head=structured env.penalty_at_retailer_only=true agent.algorithm=smoke_canon
  run "train: belief plug-in (ewma, ablation #8)" "$PY" agents/train_draco_v4.py $C agent.actor_head=structured agent.belief_mode=ewma agent.algorithm=smoke_belief
  run "train: IPPO independent critic (#19)"       "$PY" agents/train_draco_v4.py $C agent.actor_head=structured agent.critic_mode=independent agent.algorithm=smoke_ippo

  CKPT="$(ls -t weights_draco/*/draco_checkpoint_best.pt 2>/dev/null | head -1)"
  if [ -n "$CKPT" ]; then
    echo "   (using checkpoint: $CKPT)"
    run "eval: standard + regime(C1) + families + bullwhip"  "$PY" agents/eval_draco_v4.py --ckpt "$CKPT" --episodes 3 --regime-episodes 2 --families --family-episodes 2 --ar1-rhos 0.0 0.6 --bullwhip --bullwhip-episodes 2
    run "eval: dump-c1 producer"                  "$PY" agents/eval_draco_v4.py --ckpt "$CKPT" --dump-c1 results/smoke_c1 --dump-c1-episodes 2 --seed 0
    run "c1_stats report (needs refs json)"       bash -c "[ -f results/baselines_regime_v2.json ] && \"$PY\" scripts/c1_stats.py report --draco-dir results/smoke_c1 || echo '(skip: run baselines.py regime first)'"
    run "distill on real ckpt (linear, 1 DAgger round)" "$PY" scripts/distill_symbolic.py --ckpt "$CKPT" --backend linear --dagger-rounds 1 --bc-episodes 3 --dagger-episodes 2 --eval-episodes 3 --fidelity-episodes 2
  else
    echo "   FAIL: no checkpoint produced -> eval/distill skipped"; fail=$((fail+1))
  fi
else
  echo ">> torch/wandb/hydra not importable -> SKIPPING agent/training/eval/distill stages"
  echo "   (run this on the pod, or in a venv with the full requirements, to smoke those.)"
  skip=$((skip+1))
fi

echo "============================================================"
echo "SMOKE SUMMARY:  PASS=$pass  FAIL=$fail  SKIPPED(groups)=$skip"
echo "============================================================"
[ "$fail" -eq 0 ]
