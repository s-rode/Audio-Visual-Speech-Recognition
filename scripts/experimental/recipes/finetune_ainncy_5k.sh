#!/usr/bin/env bash
set -euo pipefail

# Larger interim MMS-LLaMA fine-tuning recipe using the Ainncy/LRS3 HF mirror.
# Train/eval are produced by the same local AV-HuBERT preprocessing path.

RAW_DIR=${RAW_DIR:-data/lrs3/ainncy_ft_5500_raw}
AVHUBERT_DIR=${AVHUBERT_DIR:-data/lrs3/ainncy_ft_5500_avhubert}
MMS_DIR=${MMS_DIR:-data/lrs3/mms_finetune_ainncy_5000_500}
OUT_DIR=${OUT_DIR:-outputs/mms_llama_finetune_ainncy_5000_500}

EXPORT_COUNT=${EXPORT_COUNT:-5500}
TRAIN_COUNT=${TRAIN_COUNT:-5000}
VALID_COUNT=${VALID_COUNT:-500}
SEED=${SEED:-1}

uv run team1-prepare-lrs3-hf \
  "$RAW_DIR" \
  --limit "$EXPORT_COUNT" \
  --seed "$SEED" \
  --keep-video \
  --force

LD_LIBRARY_PATH="/opt/cuda/lib64:/usr/local/cuda/lib64:/usr/lib:${LD_LIBRARY_PATH:-}" \
uv run team1-preprocess-avhubert-video \
  "$RAW_DIR/manifest.csv" \
  "$AVHUBERT_DIR" \
  --cnn-detector external/assets/dlib/mmod_human_face_detector.dat \
  --face-predictor external/assets/dlib/shape_predictor_68_face_landmarks.dat \
  --mean-face external/assets/dlib/20words_mean_face.npy \
  --force

python scripts/experimental/build_mms_llama_finetune_manifest.py \
  "$AVHUBERT_DIR/manifest.csv" \
  "$MMS_DIR" \
  --train-count "$TRAIN_COUNT" \
  --valid-count "$VALID_COUNT" \
  --seed "$SEED"

MAX_UPDATE=${MAX_UPDATE:-5000} \
MAX_EPOCH=${MAX_EPOCH:-20} \
MAX_TOKENS=${MAX_TOKENS:-500} \
UPDATE_FREQ=${UPDATE_FREQ:-8} \
LR=${LR:-5e-5} \
SAVE_INTERVAL_UPDATES=${SAVE_INTERVAL_UPDATES:-500} \
scripts/experimental/train_mms_llama_single_gpu.sh \
  "$MMS_DIR" \
  "$OUT_DIR"
