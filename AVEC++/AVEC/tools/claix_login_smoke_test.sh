#!/usr/bin/zsh
# Quick sanity check for the REAL CLAIX config (configs/EffConfInterCTC_pretrained_claix.py),
# meant to run directly on the login23-g-1 node (the only login node with GPUs) --
# NOT via sbatch. Point of this is to catch config/data-path mistakes (wrong
# data_root, wrong version/mode, missing checkpoint files, etc.) in a minute or
# two instead of finding out after waiting in the Slurm queue.
#
# Usage:
#   ssh <you>@login23-g-1.hpc.itc.rwth-aachen.de
#   cd .../team-1/AVEC++/AVEC   # wherever you rsynced AVEC++/AVEC to on CLAIX
#   ./tools/claix_login_smoke_test.sh
#
# Login-node limits this script deliberately stays under (see
# https://help.itc.rwth-aachen.de/en/service/rhr4fjjutttf/article/0a23d513f31b4cf1849986aaed475789/):
#   - up to 4 CPU cores per user
#   - GPU use capped at 30 min per process (auto-killed past that)
#   - processes using significant CPU for >20 min may get killed without warning
#   - your processes combined capped at 25% of node memory
# `timeout` below cuts the run at 20 min so Slurm's own 30-min GPU cutoff never
# has to be the thing that kills it.

set -e

if [[ ! -f main.py ]]; then
    echo "main.py not found in $(pwd) -- run this from inside AVEC++/AVEC/" >&2
    exit 1
fi

source .venv/bin/activate   # plain venv built from requirements-claix.txt (see CLAIX_SETUP.md)

echo "Running a capped smoke pass: steps_per_epoch=20, 4 workers, 20 min hard timeout."
echo "Watch for: dataset paths resolving, SwinLip/wav2vec2 loading, a few logged"
echo "training steps, and a checkpoint written under callback_path. Ctrl-C once"
echo "you've seen a couple of log lines if you don't want to wait for the timeout."

timeout 20m python main.py \
    --config_file configs/EffConfInterCTC_pretrained_claix.py \
    --mode training \
    --steps_per_epoch 20 \
    --step_log_period 5 \
    --saving_period_epoch 1 \
    -j 4

echo "Smoke test finished (or timed out) -- check the output above and"
echo "callback_path/logs for a checkpoint before submitting the real sbatch job."
