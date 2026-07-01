import os
from pathlib import Path
import sys
sys.path.append("../../")

import nnet
import torch


def _env_or_existing_path(env_name, local_path, cluster_path):
    value = os.environ.get(env_name)
    if value:
        return value
    if Path(local_path).exists():
        return local_path
    return cluster_path


# Architecture, matching the released LRS23 AO checkpoint.
vocab_size = 256
interctc_blocks = []
loss_weights = None
att_type = "patch"

# Paths
tokenizer_path = "datasets/LRS3/tokenizerbpe256.model"
pretrained_checkpoint = os.environ.get(
    "AVEC_AUDIO_CHECKPOINT",
    "callbacks/LRS23/AO/EffConfCTC/checkpoints_swa-equal-200-210.ckpt",
)
manifest_train = _env_or_existing_path(
    "VOXMM_TRAIN_MANIFEST",
    "../../data/voxmm_preprocessed/manifest.train_no_dev_ids.local.csv",
    "/hpcwork/ww202463/team-1-data/lrs_voxmm/manifest_train_no_dev_ids/manifest.csv",
)
manifest_eval = _env_or_existing_path(
    "VOXMM_EVAL_MANIFEST",
    "../../data/voxmm_dev_eval/manifest.local.csv",
    "/hpcwork/ww202463/team-1-data/lrs_voxmm/manifest_dev_eval_avec_roi/manifest.csv",
)
callback_path = os.environ.get("VOXMM_CALLBACK_PATH", "callbacks/VoxMM/AO/EffConfCTC_finetune_dev_eval")

# Greedy CTC, matching the AV finetune runs.
beam_search = False

# Training
batch_size = int(os.environ.get("VOXMM_BATCH_SIZE", "8"))
eval_batch_size = int(os.environ.get("VOXMM_EVAL_BATCH_SIZE", str(batch_size)))
accumulated_steps = int(os.environ.get("VOXMM_ACCUMULATED_STEPS", "8"))
eval_training = False
precision = torch.float32
recompute_metrics = True
epochs = int(os.environ.get("VOXMM_EPOCHS", "15"))

# Model
model = nnet.AudioEfficientConformerInterCTC(
    vocab_size=vocab_size,
    att_type=att_type,
    interctc_blocks=interctc_blocks,
)
optimizer = nnet.Adam(
    params=model.parameters(),
    lr=nnet.CosineAnnealingScheduler(
        warmup_steps=int(os.environ.get("VOXMM_WARMUP_STEPS", "300")),
        val_max=float(os.environ.get("VOXMM_PEAK_LR", "1e-4")),
        val_min=float(os.environ.get("VOXMM_MIN_LR", "1e-6")),
        end_step=int(os.environ.get("VOXMM_SCHEDULER_STEPS", "9000")),
    ),
    betas=(0.9, 0.98),
    eps=1e-9,
    weight_decay=1e-6,
)
model.compile(
    losses=nnet.CTCLoss(zero_infinity=True, assert_shorter=False),
    metrics=nnet.WordErrorRate(),
    decoders=nnet.CTCGreedySearchDecoder(tokenizer_path=tokenizer_path),
    loss_weights=loss_weights,
    optimizer=optimizer,
)

checkpoint = torch.load(pretrained_checkpoint, map_location=model.device)
model.load_state_dict(checkpoint["model_state_dict"], strict=True)
print("Loaded pretrained audio checkpoint:", pretrained_checkpoint)

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Trainable params: {trainable:,}")

# W&B logging
import wandb

wandb.init(
    project=os.environ.get("WANDB_PROJECT", "team1-asr"),
    name=os.environ.get("WANDB_RUN_NAME", "avec-voxmm-audio-only-finetune"),
    config={
        "model": "AudioEfficientConformerInterCTC",
        "pretrained": pretrained_checkpoint,
        "train_manifest": manifest_train,
        "eval_manifest": manifest_eval,
        "trainable_params": trainable,
        "batch_size": batch_size,
        "accumulated_steps": accumulated_steps,
        "peak_lr": float(os.environ.get("VOXMM_PEAK_LR", "1e-4")),
        "warmup_steps": int(os.environ.get("VOXMM_WARMUP_STEPS", "300")),
        "dataset": os.environ.get("VOXMM_DATASET_NAME", "VoxMM-train-no-dev-ids"),
    },
)

_original_log_step = model.log_step.__func__


def _wandb_log_step(self, losses, metrics, infos, writer, step, tag, dist_log=False):
    _original_log_step(self, losses, metrics, infos, writer, step, tag, dist_log)
    if self.rank == 0:
        log_dict = {f"{tag}/{k}": v for k, v in losses.items()}
        log_dict.update({f"{tag}/{k}": v for k, v in metrics.items()})
        log_dict.update({f"{tag}/{k}": float(v) for k, v in infos.items() if isinstance(v, (float, int))})
        wandb.log(log_dict, step=int(self.model_step))


import types
model.log_step = types.MethodType(_wandb_log_step, model)

# Datasets
audio_max_length = 16 * 16000
collate_fn = nnet.CollateFn(
    inputs_params=[{"axis": 1, "padding": True}, {"axis": 4}],
    targets_params=({"axis": 2, "padding": True}, {"axis": 5}),
)

training_dataset = nnet.datasets.VoxMM(
    batch_size=batch_size,
    collate_fn=collate_fn,
    manifest_path=manifest_train,
    tokenizer_path=tokenizer_path,
    audio_max_length=audio_max_length,
    load_audio=True,
    load_video=False,
    align=False,
)

evaluation_dataset = nnet.datasets.VoxMM(
    batch_size=eval_batch_size,
    collate_fn=collate_fn,
    manifest_path=manifest_eval,
    tokenizer_path=tokenizer_path,
    load_audio=True,
    load_video=False,
    align=False,
)
