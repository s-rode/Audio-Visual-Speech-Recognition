#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
decoder_python="${VOXMM_DECODER_PYTHON:-$repo_root/external/AVEC/.venv/bin/python}"
lm_python="${VOXMM_LM_PYTHON:-$repo_root/external/MMS-LLaMA/.venv/bin/python}"
cache_dir="${VOXMM_EMISSION_CACHE:-$repo_root/outputs/voxmm_emission_cache/epoch12}"
eval_manifest="${VOXMM_EVAL_MANIFEST:-$repo_root/data/voxmm_dev_eval/manifest.local.csv}"
train_manifest="${VOXMM_TRAIN_MANIFEST:-$repo_root/data/voxmm_preprocessed/manifest.train_no_dev_ids.local.csv}"
kenlm_path="${VOXMM_KENLM_PATH:-$repo_root/external/assets/lm/voxmm-3gram-d060/3-gram.arpa}"
lexicon_unigrams="${VOXMM_LEXICON_UNIGRAM_PATH:-$repo_root/external/assets/lm/librispeech-4gram/unigrams.txt}"
output_dir="${OUTPUT_DIR:-$repo_root/outputs/voxmm_roberta_rescore}"
nbest_dir="$output_dir/nbest"
nbest_name="domain_beam500"
generated_nbest="$nbest_dir/${nbest_name}_nbest.jsonl"
candidate_path="${VOXMM_NBEST_PATH:-$generated_nbest}"
top_k="${VOXMM_RERANK_TOP_K:-50}"
masked_model="${VOXMM_MASKED_LM:-FacebookAI/roberta-base}"
scored_path="$output_dir/roberta_top${top_k}_scored.jsonl"
force="${VOXMM_FORCE:-0}"

for executable in "$decoder_python" "$lm_python"; do
  if [[ ! -x "$executable" ]]; then
    echo "Missing Python executable: $executable" >&2
    exit 1
  fi
done

if [[ ! -f "$candidate_path" ]]; then
  for path in "$eval_manifest" "$train_manifest" "$kenlm_path" "$lexicon_unigrams"; do
    if [[ ! -f "$path" ]]; then
      echo "Missing required input: $path" >&2
      exit 1
    fi
  done
  if ! compgen -G "$cache_dir/batch_*.pt" >/dev/null; then
    echo "No cached emissions found under $cache_dir" >&2
    exit 1
  fi

  "$decoder_python" "$repo_root/scripts/experimental/sweep_voxmm_cached_decoder.py" \
    "$cache_dir" \
    --manifest "$eval_manifest" \
    --lm-manifest "$train_manifest" \
    --kenlm "$kenlm_path" \
    --lexicon-unigrams "$lexicon_unigrams" \
    --max-words 200000 \
    --nbest 100 \
    --save-nbest \
    --setting "$nbest_name:1.15:1.10:500:50:50:false:1.0" \
    --output-dir "$nbest_dir"
fi

forward_checkpoint="${VOXMM_FORWARD_LM_CHECKPOINT:-}"
reverse_checkpoint="${VOXMM_REVERSE_LM_CHECKPOINT:-}"
if [[ -n "$forward_checkpoint" ]]; then
  annotated_dir="$output_dir/compact_lm"
  annotated_path="$annotated_dir/scored_nbest.jsonl"
  if [[ "$force" == 1 || ! -f "$annotated_path" ]]; then
    compact_args=(
      rescore "$candidate_path" "$forward_checkpoint" "$annotated_dir"
      --setting annotate:0:0
    )
    if [[ -n "$reverse_checkpoint" ]]; then
      compact_args+=(--aux-checkpoint "$reverse_checkpoint")
    fi
    "$lm_python" \
      "$repo_root/scripts/experimental/voxmm_neural_lm.py" \
      "${compact_args[@]}"
  fi
  candidate_path="$annotated_path"
fi

mkdir -p "$output_dir"
export HF_HOME="${HF_HOME:-$repo_root/.tmp/hf}"
if [[ "$force" == 1 || ! -f "$scored_path" ]]; then
  score_args=(
    score "$candidate_path" "$scored_path"
    --backend mlm
    --masked-model "$masked_model"
    --top-k "$top_k"
    --drop-filled-pauses
    --batch-size "${VOXMM_MLM_BATCH_SIZE:-64}"
  )
  if [[ -n "${VOXMM_MLM_DEVICE:-}" ]]; then
    score_args+=(--device "$VOXMM_MLM_DEVICE")
  fi
  "$lm_python" \
    "$repo_root/scripts/experimental/voxmm_transformer_lm.py" \
    "${score_args[@]}"
fi

transformer_weight="${VOXMM_TRANSFORMER_WEIGHT:-0.16}"
word_bonus="${VOXMM_SECOND_PASS_WORD_BONUS:-1.30}"
forward_weight="${VOXMM_FORWARD_LM_WEIGHT:-0.0}"
reverse_weight="${VOXMM_REVERSE_LM_WEIGHT:-0.0}"
# Reward pause-bearing acoustic paths slightly; output normalization removes
# those pauses before scoring against VoxMM's pause-free references.
drop_penalty="${VOXMM_DROP_PENALTY:--1.0}"
anchor="$transformer_weight:$word_bonus:$forward_weight:$reverse_weight:$drop_penalty"

if [[ "${VOXMM_TUNE_WEIGHTS:-0}" == 1 ]]; then
  "$lm_python" \
    "$repo_root/scripts/experimental/tune_voxmm_transformer_lm.py" \
    "$scored_path" "$output_dir/tuned" \
    --top-k "$top_k" \
    --drop-filled-pauses \
    --samples "${VOXMM_TUNE_SAMPLES:-50000}" \
    --rounds "${VOXMM_TUNE_ROUNDS:-3}" \
    --anchor "$anchor"
else
  setting="best:$transformer_weight:$word_bonus:1:$drop_penalty:0:$forward_weight:$reverse_weight"
  "$lm_python" \
    "$repo_root/scripts/experimental/voxmm_transformer_lm.py" \
    rescore "$scored_path" "$output_dir/fixed" \
    --drop-filled-pauses \
    --setting "$setting"
fi
