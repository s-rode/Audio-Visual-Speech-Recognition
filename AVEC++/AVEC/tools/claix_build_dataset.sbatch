#!/usr/bin/zsh
# Build one LRS2-shaped split (train or val) from a manifest CSV via
# tools/build_lrs2_from_manifest.py. CPU-only preprocessing (video read +
# audio resample + tokenize per row) -- no GPU needed, hence c23ms not c23g.
#
# Submit with:
#   sbatch tools/claix_build_dataset.sbatch <manifest.csv> <split> [--no-skip-existing]
# e.g. (run both -- safe to do concurrently, they write disjoint utterance
# files and separate <split>.txt filelists, no shared state between them):
#   sbatch tools/claix_build_dataset.sbatch /rwthfs/rz/cluster/hpcwork/lect0176/team-1/data/lrs3/avec_roi/manifest_train.csv train
#   sbatch tools/claix_build_dataset.sbatch /rwthfs/rz/cluster/hpcwork/lect0176/team-1/data/lrs3/avec_roi/manifest_val.csv val
#
# Safe to resubmit the SAME (manifest, split) pair if a run times out or gets
# killed: build_lrs2_from_manifest.py's --skip-existing (on by default) means
# a re-run resumes rather than redoes finished utterances, and still writes a
# complete/correct <split>.txt at the end since it re-reads the full manifest
# each time.
#
# Pass --no-skip-existing as the optional 3rd arg to force every utterance to
# be rebuilt instead of reusing existing .pt files -- required any time the
# tokenizer changes (old .pt labels were encoded with the old vocab and won't
# fix themselves) or the train/val split changes (an utterance that moved
# sides still has a .pt sitting under the shared main/ dir from before; the
# split membership itself comes from which manifest you pass in and always
# gets applied correctly, but the label encoding won't be refreshed without
# this flag). Both applied at once here: e.g.
#   sbatch tools/claix_build_dataset.sbatch manifest_train.csv train --no-skip-existing

### Job Parameters
#SBATCH --job-name=avecpp-build
#SBATCH --account=lect0176               # confirm with `sacctmgr show assoc user=$USER` if unsure
#SBATCH --partition=c23ms                # CPU only, no GPU needed for this step
#SBATCH --cpus-per-task=1                # build_lrs2_from_manifest.py is single-threaded
#SBATCH --time=12:00:00                  # rough guess for ~25k rows -- unverified, resubmit if it's not enough
#SBATCH --output=logs/%x_%j.out          # mkdir -p logs first (Slurm won't create it)

### Program Code
if [[ ! -f main.py ]]; then
    echo "main.py not found in $(pwd) -- submit this from inside AVEC++/AVEC/" >&2
    exit 1
fi
if [[ $# -lt 2 ]]; then
    echo "Usage: sbatch tools/claix_build_dataset.sbatch <manifest.csv> <split> [--no-skip-existing]" >&2
    exit 1
fi

MANIFEST="$1"
SPLIT="$2"
EXTRA_FLAG="${3:-}"

source .venv/bin/activate
mkdir -p logs

python tools/build_lrs2_from_manifest.py \
    --manifest "$MANIFEST" \
    --split "$SPLIT" \
    --tokenizer ../tokenizer/tokenizerbpe256.model \
    $EXTRA_FLAG
