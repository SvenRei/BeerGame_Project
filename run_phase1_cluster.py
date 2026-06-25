import os
import sys
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed

# --- 1. SAFE SETUP CHECKS ---
if "WANDB_API_KEY" not in os.environ:
    print("ERROR: WANDB_API_KEY is not set. Please run 'wandb login' or export it.")
    sys.exit(1)

if not os.path.exists("agents/train_draco_v4.py"):
    print("ERROR: Must run from the root of the BeerGame_Project repository.")
    sys.exit(1)

os.makedirs("run_logs", exist_ok=True)

# --- 2. DYNAMIC SEEDS ---
# Matches the "$@" behavior of the bash script: requires explicit seed args
if len(sys.argv) > 1:
    SEEDS = sys.argv[1:]
else:
    print("ERROR: Please provide seeds as arguments (e.g., python run_phase1_cluster.py 0 1 2)")
    sys.exit(1)

# --- SET YOUR MAX WORKERS ---
MAX_CONCURRENT_RUNS = 14

AUX_COEFS = [0.1, 0.3, 0.5]
Z_DIMS = [4, 8, 16]
ENCODERS = ["gru", "craft"]

VAL = "agent.heldout_lambdas=[8,12,16,20]"

# MATCHED PARITY: batch_episodes=8 exactly matches your updated phase1_sweep.sh
BASE = (
    "agent=draco_v4 agent.use_comm=false agent.actor_head=structured "
    "agent.use_context=true agent.dr_lambda_lo=4 agent.dr_lambda_hi=24 "
    "agent.dr_p_shift=0.0 agent.heldout_every=400 agent.heldout_episodes=20 "
    "total_episodes=15000 agent.batch_episodes=8 agent.patience=3000 " + VAL
)

def run_experiment(cmd_tuple):
    """Executes a command and writes output to a unique log file to prevent hidden failures."""
    algo_name, cmd = cmd_tuple
    log_file = f"run_logs/{algo_name}.log"
    
    with open(log_file, "w") as f:
        try:
            # check=True ensures we catch non-zero exit codes
            subprocess.run(cmd, shell=True, stdout=f, stderr=subprocess.STDOUT, check=True)
            return algo_name, True, None
        except subprocess.CalledProcessError:
            return algo_name, False, log_file

if __name__ == "__main__":
    commands = []
    for seed in SEEDS:
        for aux in AUX_COEFS:
            for z in Z_DIMS:
                for enc in ENCODERS:
                    algo_name = f"p1_aux{aux}_z{z}_{enc}_s{seed}"
                    cmd = (
                        f"python agents/train_draco_v4.py {BASE} "
                        f"agent.demand_aux_coef={aux} agent.z_dim={z} "
                        f"agent.encoder_type={enc} seed={seed} "
                        f"agent.algorithm={algo_name}"
                    )
                    commands.append((algo_name, cmd))

    print(f"Queueing {len(commands)} runs across {MAX_CONCURRENT_RUNS} parallel workers...")
    
    with ProcessPoolExecutor(max_workers=MAX_CONCURRENT_RUNS) as executor:
        futures = {executor.submit(run_experiment, cmd_tuple): cmd_tuple[0] for cmd_tuple in commands}
        
        for i, future in enumerate(as_completed(futures), 1):
            name, success, log_path = future.result()
            if success:
                print(f"[{i}/{len(commands)}] SUCCESS: {name}")
            else:
                print(f"[{i}/{len(commands)}] FAILED:  {name} (Check {log_path} for errors)")

    print("Phase 1 Cluster Sweep Complete!")