import os
import sys
import time
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed

if not os.path.exists("agents/train_draco_v4.py"):
    print("ERROR: Must run from the root of the BeerGame_Project repository.")
    sys.exit(1)
os.makedirs("run_logs", exist_ok=True)

# --- 2. DYNAMIC SEEDS ---
if len(sys.argv) > 1:
    SEEDS = sys.argv[1:]
else:
    print("ERROR: provide seeds as args (e.g., python run_phase1_cluster.py 0 1 2)")
    sys.exit(1)

# --- 3. PARALLELISM (this box: 32 vCPU, 1 GPU) ---
MAX_CONCURRENT_RUNS = 10          # start at 10; raise toward 14 only after watching nvidia-smi/htop
N_VCPU = 32
THREADS_PER_RUN = max(1, N_VCPU // MAX_CONCURRENT_RUNS)   # cap torch threads -> avoid 450-thread thrash
USE_GPU = True                    # set False to run all on CPU (tiny models -> often fine, zero GPU contention)
STAGGER_SECONDS = 3.0             # delay between launches so 10 CUDA contexts don't allocate at once

AUX_COEFS = [0.1, 0.3, 0.5]
Z_DIMS = [4, 8, 16]
ENCODERS = ["gru", "craft"]
VAL = "agent.heldout_lambdas=[8,12,16,20]"

# MATCHED PARITY with phase1_sweep.sh (batch_episodes=8, total_episodes=15000, same VAL split)
BASE = (
    "agent=draco_v4 agent.use_comm=false agent.actor_head=structured "
    "agent.use_context=true agent.dr_lambda_lo=4 agent.dr_lambda_hi=24 "
    "agent.dr_p_shift=0.0 agent.heldout_every=400 agent.heldout_episodes=20 "
    "total_episodes=15000 agent.batch_episodes=8 agent.patience=3000 " + VAL
)


def run_experiment(args):
    algo_name, cmd, stagger = args
    if stagger:
        time.sleep(stagger)       # spread CUDA-context allocation across the first wave
    env = dict(os.environ)
    # cap intra-op threads PER PROCESS so 10 runs don't each grab all 32 cores
    env["OMP_NUM_THREADS"] = str(THREADS_PER_RUN)
    env["MKL_NUM_THREADS"] = str(THREADS_PER_RUN)
    env["OPENBLAS_NUM_THREADS"] = str(THREADS_PER_RUN)
    env["NUMEXPR_NUM_THREADS"] = str(THREADS_PER_RUN)
    if not USE_GPU:
        env["CUDA_VISIBLE_DEVICES"] = ""     # force CPU for every run
    log_file = f"run_logs/{algo_name}.log"
    with open(log_file, "w") as f:
        try:
            subprocess.run(cmd, shell=True, stdout=f, stderr=subprocess.STDOUT, check=True, env=env)
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
                    cmd = (f"python agents/train_draco_v4.py {BASE} "
                           f"agent.demand_aux_coef={aux} agent.z_dim={z} "
                           f"agent.encoder_type={enc} seed={seed} "
                           f"agent.algorithm={algo_name}")
                    commands.append((algo_name, cmd))

    # only the first wave needs staggering (later launches start as workers free up)
    payload = [(n, c, (STAGGER_SECONDS * i if i < MAX_CONCURRENT_RUNS else 0.0))
               for i, (n, c) in enumerate(commands)]

    print(f"Queueing {len(commands)} runs | {MAX_CONCURRENT_RUNS} workers | "
          f"{THREADS_PER_RUN} threads/run | {'GPU' if USE_GPU else 'CPU'}")
    failures = []
    with ProcessPoolExecutor(max_workers=MAX_CONCURRENT_RUNS) as ex:
        futures = {ex.submit(run_experiment, p): p[0] for p in payload}
        for i, fut in enumerate(as_completed(futures), 1):
            name, ok, log_path = fut.result()
            print(f"[{i}/{len(commands)}] {'SUCCESS' if ok else 'FAILED '}: {name}"
                  + ("" if ok else f"  (see {log_path})"))
            if not ok:
                failures.append(name)

    print("\nPhase 1 Cluster Sweep Complete!"
          + (f"  {len(failures)} FAILED: {failures}" if failures else "  (all succeeded)"))