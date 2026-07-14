#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
avec_dir="$repo_root/external/AVEC"
out_dir="$repo_root/outputs/voxmm_decoder_sweep"
mkdir -p "$out_dir"

checkpoint="${CHECKPOINT:-checkpoints_epoch_12_step_7201.ckpt}"
config="${CONFIG:-configs/VoxMM/EffConfInterCTC_finetune.py}"
eval_steps="${EVAL_STEPS:-50}"
batch_size="${VOXMM_EVAL_BATCH_SIZE:-2}"
train_manifest="${VOXMM_TRAIN_MANIFEST:-../../data/voxmm_preprocessed/manifest.train_no_dev_ids.local.csv}"
eval_manifest="${VOXMM_EVAL_MANIFEST:-../../data/voxmm_dev_eval/manifest.local.csv}"
callback_path="${VOXMM_CALLBACK_PATH:-callbacks/VoxMM/AV/EffConfInterCTC_finetune_dev_eval}"
results="$out_dir/results_eval_steps_${eval_steps}.tsv"

printf "name\tdecoder\torder\tlm_weight\tword_bonus\tbeam\tcutoff\ttemp\ttoken_bonus\twer\n" > "$results"

run_eval() {
    local name="$1"
    local decoder="$2"
    local order="$3"
    local lm_weight="$4"
    local word_bonus="$5"
    local beam="$6"
    local cutoff="$7"
    local temp="$8"
    local token_bonus="$9"
    local log_file="$out_dir/${name}_eval_steps_${eval_steps}.log"

    echo "Running $name"
    (
        cd "$avec_dir"
        HOME="$repo_root/external/.home" \
        XDG_CACHE_HOME="$repo_root/external/.cache" \
        MPLCONFIGDIR="$repo_root/external/.cache/matplotlib" \
        WANDB_MODE=disabled \
        VOXMM_DECODER="$decoder" \
        VOXMM_LM_MANIFEST="$train_manifest" \
        VOXMM_LM_TYPE=kneser_ney \
        VOXMM_LM_ORDER="$order" \
        VOXMM_LM_WEIGHT="$lm_weight" \
        VOXMM_WORD_BONUS="$word_bonus" \
        VOXMM_BEAM_SIZE="$beam" \
        VOXMM_BEAM_CUTOFF_TOP_N="$cutoff" \
        VOXMM_BEAM_TEMPERATURE="$temp" \
        VOXMM_BEAM_TOKEN_BONUS="$token_bonus" \
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
    printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
        "$name" "$decoder" "$order" "$lm_weight" "$word_bonus" "$beam" "$cutoff" "$temp" "$token_bonus" "$wer" >> "$results"
}

run_eval greedy greedy 4 0.0 0.0 1 1 1.0 0.0
run_eval beam_no_lm_12 prefix_beam 4 0.0 0.0 12 30 1.0 0.0
run_eval beam_no_lm_24 prefix_beam 4 0.0 0.0 24 40 1.0 0.0
run_eval kn4_w003_b05 prefix_beam_lm 4 0.03 0.5 12 30 1.0 0.0
run_eval kn4_w005_b05 prefix_beam_lm 4 0.05 0.5 12 30 1.0 0.0
run_eval kn4_w007_b05 prefix_beam_lm 4 0.07 0.5 12 30 1.0 0.0
run_eval kn4_w010_b05 prefix_beam_lm 4 0.10 0.5 12 30 1.0 0.0
run_eval kn4_w007_b00 prefix_beam_lm 4 0.07 0.0 12 30 1.0 0.0
run_eval kn4_w007_b10 prefix_beam_lm 4 0.07 1.0 12 30 1.0 0.0
run_eval kn3_w007_b05 prefix_beam_lm 3 0.07 0.5 12 30 1.0 0.0
run_eval kn5_w007_b05 prefix_beam_lm 5 0.07 0.5 12 30 1.0 0.0
run_eval kn4_w007_b05_beam24 prefix_beam_lm 4 0.07 0.5 24 40 1.0 0.0
run_eval kn4_w007_b05_temp08 prefix_beam_lm 4 0.07 0.5 12 30 0.8 0.0
run_eval kn4_w007_b05_temp12 prefix_beam_lm 4 0.07 0.5 12 30 1.2 0.0

echo "Wrote $results"
