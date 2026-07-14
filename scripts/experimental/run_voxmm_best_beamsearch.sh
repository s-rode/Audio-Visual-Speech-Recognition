#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
avec_dir="$repo_root/external/AVEC"
train_manifest="${VOXMM_TRAIN_MANIFEST:-$repo_root/data/voxmm_preprocessed/manifest.train_no_dev_ids.local.csv}"
eval_manifest="${VOXMM_EVAL_MANIFEST:-$repo_root/data/voxmm_dev_eval/manifest.local.csv}"
lm_path="${VOXMM_KENLM_PATH:-$repo_root/external/assets/lm/voxmm-3gram-d060/3-gram.arpa}"
lexicon_is_default=0
if [[ -n "${VOXMM_LEXICON_UNIGRAM_PATH:-}" ]]; then
  lexicon_unigrams="$VOXMM_LEXICON_UNIGRAM_PATH"
else
  lexicon_unigrams="$repo_root/external/assets/lm/librispeech-4gram/unigrams.txt"
  lexicon_is_default=1
fi
callback_path="${VOXMM_CALLBACK_PATH:-callbacks/VoxMM/AV/EffConfInterCTC_finetune_dev_eval}"
checkpoint="${CHECKPOINT:-checkpoints_epoch_12_step_7201.ckpt}"
config="${CONFIG:-configs/VoxMM/EffConfInterCTC_finetune.py}"
batch_size="${VOXMM_EVAL_BATCH_SIZE:-8}"
out_dir="${OUTPUT_DIR:-$repo_root/outputs/voxmm_best_beamsearch}"
asset_dir="$out_dir/flashlight_assets"
log_file="$out_dir/evaluation.log"

for path in "$train_manifest" "$eval_manifest"; do
  if [[ ! -f "$path" ]]; then
    echo "Missing manifest: $path" >&2
    exit 1
  fi
done

if [[ ! -f "$lexicon_unigrams" ]]; then
  if (( lexicon_is_default )); then
    mkdir -p "$(dirname "$lexicon_unigrams")"
    lexicon_url="https://huggingface.co/patrickvonplaten/wav2vec2-base-960h-4-gram/resolve/996832fafdf4563dfe5a59e9338a9355891e5d59/language_model/unigrams.txt"
    echo "Downloading expanded LibriSpeech lexicon..."
    if command -v curl >/dev/null 2>&1; then
      curl --fail --location --retry 3 --output "$lexicon_unigrams" "$lexicon_url"
    elif command -v wget >/dev/null 2>&1; then
      wget -O "$lexicon_unigrams" "$lexicon_url"
    else
      echo "Neither curl nor wget is available to download $lexicon_url" >&2
      exit 1
    fi
  else
    echo "Missing expanded lexicon: $lexicon_unigrams" >&2
    exit 1
  fi
fi

if (( lexicon_is_default )); then
  expected_sha256="e77a6f780b26d2b778635496a141f0d7b1c7d22e9a7f12e7a5966c1e7a0fd072"
  actual_sha256="$(sha256sum "$lexicon_unigrams" | awk '{print $1}')"
  if [[ "$actual_sha256" != "$expected_sha256" ]]; then
    echo "Unexpected checksum for $lexicon_unigrams" >&2
    exit 1
  fi
fi

if [[ ! -f "$lm_path" ]]; then
  python3 "$repo_root/scripts/experimental/build_voxmm_ngram_arpa.py" \
    "$train_manifest" "$lm_path" \
    --order 3 \
    --discount 0.60 \
    --mode interpolated
fi

mkdir -p "$out_dir"
eval_args=()
if [[ -n "${EVAL_STEPS:-}" ]]; then
  eval_args=(--eval_steps "$EVAL_STEPS")
fi

cd "$avec_dir"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-avec-voxmm-beamsearch-best}" \
VOXMM_DECODER=flashlight_kenlm \
VOXMM_KENLM_PATH="$lm_path" \
VOXMM_UNIGRAM_PATH= \
VOXMM_LEXICON_UNIGRAM_PATH="$lexicon_unigrams" \
VOXMM_KENLM_MAX_WORDS=200000 \
VOXMM_LM_MANIFEST="$train_manifest" \
VOXMM_LM_TYPE=kneser_ney \
VOXMM_LM_ORDER=3 \
VOXMM_LM_DISCOUNT=0.60 \
VOXMM_LM_WEIGHT="${VOXMM_LM_WEIGHT:-1.15}" \
VOXMM_WORD_BONUS="${VOXMM_WORD_BONUS:-1.10}" \
VOXMM_BEAM_SIZE="${VOXMM_BEAM_SIZE:-200}" \
VOXMM_BEAM_CUTOFF_TOP_N="${VOXMM_BEAM_CUTOFF_TOP_N:-50}" \
VOXMM_BEAM_THRESHOLD="${VOXMM_BEAM_THRESHOLD:-50}" \
VOXMM_BEAM_LOG_ADD="${VOXMM_BEAM_LOG_ADD:-0}" \
VOXMM_BLANK_BIAS="${VOXMM_BLANK_BIAS:-0}" \
VOXMM_NBEST=1 \
VOXMM_RESCORE_LM_WEIGHT=0 \
VOXMM_RESCORE_WORD_BONUS=0 \
VOXMM_FLASHLIGHT_ASSET_DIR="$asset_dir" \
VOXMM_CALLBACK_PATH="$callback_path" \
VOXMM_TRAIN_MANIFEST="$train_manifest" \
VOXMM_EVAL_MANIFEST="$eval_manifest" \
VOXMM_BATCH_SIZE="$batch_size" \
VOXMM_EVAL_BATCH_SIZE="$batch_size" \
.venv/bin/python main.py \
  -c "$config" \
  --mode evaluation \
  --checkpoint "$checkpoint" \
  "${eval_args[@]}" 2>&1 | tee "$log_file"
