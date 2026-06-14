#!/usr/bin/env bash
set -euo pipefail

ROOT=${MMS_LLAMA_ROOT:-external/MMS-LLaMA}
ROOT=$(cd "$ROOT" && pwd)
SRC_PTH="$ROOT/src"
FAIRSEQ_TRAIN=${FAIRSEQ_TRAIN:-$ROOT/.venv/bin/fairseq-hydra-train}

DATA_DIR=${1:?Usage: train_mms_llama_single_gpu.sh DATA_DIR OUT_DIR [CHECKPOINT]}
OUT_DIR=${2:?Usage: train_mms_llama_single_gpu.sh DATA_DIR OUT_DIR [CHECKPOINT]}
CHECKPOINT=${3:-$ROOT/pretrained_models/mms_llama/1759h/checkpoint_best.pt}
DATA_DIR=$(realpath "$DATA_DIR")
OUT_DIR=$(mkdir -p "$OUT_DIR" && cd "$OUT_DIR" && pwd)
CHECKPOINT=$(realpath "$CHECKPOINT")

LLM_PATH=${LLM_PATH:-meta-llama/Llama-3.2-3B}
NOISE_WAV=${NOISE_WAV:-$ROOT/noise/babble_noise.wav}
MAX_UPDATE=${MAX_UPDATE:-2000}
MAX_TOKENS=${MAX_TOKENS:-500}
UPDATE_FREQ=${UPDATE_FREQ:-4}
LR=${LR:-5e-5}
SAVE_INTERVAL_UPDATES=${SAVE_INTERVAL_UPDATES:-500}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
NOISE_PROB=${NOISE_PROB:-0.0}

if [[ ! -x "$FAIRSEQ_TRAIN" ]]; then
  echo "fairseq-hydra-train not found or not executable: $FAIRSEQ_TRAIN" >&2
  exit 1
fi
if [[ ! -f "$CHECKPOINT" ]]; then
  echo "checkpoint not found: $CHECKPOINT" >&2
  exit 1
fi
if [[ ! -f "$DATA_DIR/train.tsv" || ! -f "$DATA_DIR/train.wrd" ]]; then
  echo "missing train.tsv/train.wrd in $DATA_DIR" >&2
  exit 1
fi
if [[ ! -f "$DATA_DIR/test.tsv" || ! -f "$DATA_DIR/test.wrd" ]]; then
  echo "missing test.tsv/test.wrd in $DATA_DIR" >&2
  exit 1
fi

export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="$ROOT/fairseq"
export CUDA_VISIBLE_DEVICES

"$FAIRSEQ_TRAIN" \
  --config-dir "$SRC_PTH/conf" \
  --config-name mms-llama.yaml \
  task.data="$DATA_DIR" \
  task.label_dir="$DATA_DIR" \
  task.llm_path="$LLM_PATH" \
  task.noise_prob="$NOISE_PROB" \
  task.noise_wav="$NOISE_WAV" \
  hydra.run.dir="$OUT_DIR" \
  common.user_dir="$SRC_PTH" \
  common.seed=1 \
  common.empty_cache_freq=1000 \
  checkpoint.finetune_from_model="$CHECKPOINT" \
  checkpoint.save_interval_updates="$SAVE_INTERVAL_UPDATES" \
  checkpoint.keep_interval_updates=2 \
  checkpoint.no_epoch_checkpoints=true \
  model.w2v_path="$ROOT/pretrained_models/avhubert/large_vox_iter5.pt" \
  model.llm_path="$LLM_PATH" \
  model.llama_embed_dim=3072 \
  model.queries_per_sec=3 \
  model.target_modules=q_proj.k_proj.v_proj.o_proj \
  model.modality_fuse=concat \
  model.lora_rank=16 \
  model.lora_alpha=32 \
  model.use_qformer=true \
  model.use_sr_predictor=true \
  dataset.max_tokens="$MAX_TOKENS" \
  dataset.num_workers=1 \
  optimization.update_freq="[$UPDATE_FREQ]" \
  optimization.lr="[$LR]" \
  optimization.max_update="$MAX_UPDATE" \
  lr_scheduler._name=cosine \
  lr_scheduler.warmup_updates=100 \
  distributed_training.distributed_world_size=1 \
  distributed_training.nprocs_per_node=1 \
  distributed_training.ddp_backend=legacy_ddp \
  distributed_training.find_unused_parameters=true
