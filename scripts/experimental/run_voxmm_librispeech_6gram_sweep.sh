#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
avec_dir="$repo_root/external/AVEC"
out_dir="$repo_root/outputs/voxmm_librispeech_6gram_sweep"
mkdir -p "$out_dir"

checkpoint="${CHECKPOINT:-checkpoints_epoch_12_step_7201.ckpt}"
config="${CONFIG:-configs/VoxMM/EffConfInterCTC_finetune.py}"
eval_steps="${EVAL_STEPS:-50}"
batch_size="${VOXMM_EVAL_BATCH_SIZE:-2}"
train_manifest="${VOXMM_TRAIN_MANIFEST:-../../data/voxmm_preprocessed/manifest.train_no_dev_ids.local.csv}"
eval_manifest="${VOXMM_EVAL_MANIFEST:-../../data/voxmm_dev_eval/manifest.local.csv}"
callback_path="${VOXMM_CALLBACK_PATH:-callbacks/VoxMM/AV/EffConfInterCTC_finetune_dev_eval}"
kenlm_path="${VOXMM_KENLM_PATH:-$repo_root/external/assets/lm/librispeech-6gram/6-gram.pruned.bin}"
unigram_path="${VOXMM_UNIGRAM_PATH:-$repo_root/external/assets/lm/librispeech-4gram/unigrams.txt}"
max_words="${VOXMM_KENLM_MAX_WORDS:-200000}"
results="$out_dir/results_eval_steps_${eval_steps}.tsv"

if [[ ! -f "$kenlm_path" ]]; then
  echo "Missing KenLM binary: $kenlm_path" >&2
  echo "Run scripts/experimental/build_librispeech_6gram_kenlm.sh first." >&2
  exit 1
fi

printf "name\tdecoder\tlm_weight\tword_bonus\tbeam\tcutoff\tmax_words\twer\n" > "$results"

run_eval() {
  local name="$1"
  local lm_weight="$2"
  local word_bonus="$3"
  local beam="$4"
  local cutoff="$5"
  local asset_dir="$out_dir/assets_${name}"
  local log_file="$out_dir/${name}_eval_steps_${eval_steps}.log"

  echo "Running $name"
  (
    cd "$avec_dir"
    HOME="$repo_root/external/.home" \
    XDG_CACHE_HOME="$repo_root/external/.cache" \
    MPLCONFIGDIR="$repo_root/external/.cache/matplotlib" \
    WANDB_MODE=disabled \
    VOXMM_DECODER=flashlight_kenlm \
    VOXMM_KENLM_PATH="$kenlm_path" \
    VOXMM_UNIGRAM_PATH="$unigram_path" \
    VOXMM_KENLM_MAX_WORDS="$max_words" \
    VOXMM_LM_MANIFEST="$train_manifest" \
    VOXMM_LM_WEIGHT="$lm_weight" \
    VOXMM_WORD_BONUS="$word_bonus" \
    VOXMM_BEAM_SIZE="$beam" \
    VOXMM_BEAM_CUTOFF_TOP_N="$cutoff" \
    VOXMM_FLASHLIGHT_ASSET_DIR="$asset_dir" \
    VOXMM_CALLBACK_PATH="$callback_path" \
    VOXMM_TRAIN_MANIFEST="$train_manifest" \
    VOXMM_EVAL_MANIFEST="$eval_manifest" \
    VOXMM_BATCH_SIZE="$batch_size" \
    VOXMM_EVAL_BATCH_SIZE="$batch_size" \
    .venv/bin/python main.py -c "$config" \
      --mode evaluation \
      --checkpoint "$checkpoint" \
      --eval_steps "$eval_steps"
  ) 2>&1 | tee "$log_file"

  local wer
  wer="$(awk '/eval wer:/ {value=$3} END {print value}' "$log_file")"
  printf "%s\tflashlight_kenlm\t%s\t%s\t%s\t%s\t%s\t%s\n" \
    "$name" "$lm_weight" "$word_bonus" "$beam" "$cutoff" "$max_words" "$wer" >> "$results"
}

run_eval gram6_w040_bm05 0.40 -0.5 50 50
run_eval gram6_w050_bm05 0.50 -0.5 50 50
run_eval gram6_w060_bm05 0.60 -0.5 50 50
run_eval gram6_w070_bm05 0.70 -0.5 50 50
run_eval gram6_w060_bm025 0.60 -0.25 50 50
run_eval gram6_w060_bm075 0.60 -0.75 50 50
run_eval gram6_w080_bm05 0.80 -0.5 50 50

echo "Wrote $results"
