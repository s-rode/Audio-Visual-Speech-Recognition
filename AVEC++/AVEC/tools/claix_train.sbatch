#!/usr/bin/zsh
# CLAIX (RWTH Aachen) batch script for the real AVEC++ training run.
# https://help.itc.rwth-aachen.de/en/service/rhr4fjjutttf/article/13ace46cfbb84e92a64c1361e0e4c104/
#
# Submit with:  sbatch tools/claix_train.sbatch
# Check status: squeue --me
# Cancel:       scancel <jobid>
#
# Working directory when sbatch runs is wherever you called `sbatch` from --
# submit this from inside AVEC++/AVEC/ (see CLAIX_SETUP.md for the exact path)
# so the relative paths in the config (configs/EffConfInterCTC_pretrained_claix.py)
# resolve the same way they do when you run main.py by hand.

### Job Parameters
#SBATCH --job-name=avecpp-train
#SBATCH --account=lect0176               # <-- your team's course project id, inferred from the
                                          # hpcwork path (.../hpcwork/lect0176/team-1) -- NOT your
                                          # personal username. Confirm with `sacctmgr show assoc user=$USER`
#SBATCH --partition=c23g                 # GPU partition (4x H100 per node)
#SBATCH --gres=gpu:1                     # 1 GPU = 24 cores / 122GB on that node, proportionally
#SBATCH --cpus-per-task=8                # data loading workers (see -j/--num_workers below)
#SBATCH --time=1-00:00:00                # 1 day -- raise once you know real epoch time from the smoke run
#SBATCH --output=logs/%x_%j.out          # combines stdout+stderr; Slurm won't create the dir
                                          # itself -- run `mkdir -p logs` in AVEC/ before submitting

### Best practice (per RWTH docs): the FIRST time you submit this, drop
### --time down to something like 00:05:00 so a config mistake fails fast
### instead of you waiting hours in the queue to find out. Only bump it back
### up to the real time budget once a short run has gone through cleanly.

### Program Code
# Slurm sets the working directory to wherever `sbatch` was called from --
# this script assumes that was AVEC++/AVEC/ itself.
if [[ ! -f main.py ]]; then
    echo "main.py not found in $(pwd) -- submit this from inside AVEC++/AVEC/ (see CLAIX_SETUP.md)" >&2
    exit 1
fi

module purge
# module load <whatever CLAIX requires you to load before `uv` works, if anything>

source .venv/bin/activate   # plain venv built from requirements-claix.txt (see CLAIX_SETUP.md)

mkdir -p logs

# --load_last: resumes from the newest checkpoint in callback_path if one
# exists, otherwise starts fresh (find_last_checkpoint returns None on an
# empty dir) -- safe to leave on permanently, including on the very first
# submission. This matters every time this script gets resubmitted to
# continue training (config edits, time-limit resubmits, etc.) -- without
# it, a resubmit silently restarts from epoch 0 and throws away all prior
# progress.
python main.py \
    --config_file configs/EffConfInterCTC_pretrained_claix.py \
    --mode training \
    --load_last \
    -j 8
