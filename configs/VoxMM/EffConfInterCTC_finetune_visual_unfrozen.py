import os
from pathlib import Path
import sys
sys.path.append("../../")

import nnet
import torch
import torch.nn as nn
import torchvision


def _env_or_existing_path(env_name, local_path, cluster_path):
    value = os.environ.get(env_name)
    if value:
        return value
    if Path(local_path).exists():
        return local_path
    return cluster_path


# Architecture, matching the released LRS23 AV checkpoint.
vocab_size = 256
v_interctc_blocks = [3, 6]
a_interctc_blocks = [8, 11]
f_interctc_blocks = [2]
loss_weights = {
    "v_ctc_2": 0.5 / 3,
    "v_ctc_5": 0.5 / 3,
    "a_ctc_7": 0.5 / 3,
    "a_ctc_10": 0.5 / 3,
    "f_ctc_1": 0.5 / 3,
    "outputs": 0.5,
}

# Paths
tokenizer_path = "datasets/LRS3/tokenizerbpe256.model"
pretrained_checkpoint = os.environ.get(
    "AVEC_PRETRAINED_CHECKPOINT",
    "callbacks/LRS23/AV/EffConfInterCTC/checkpoints_swa-equal-60-70.ckpt",
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
callback_path = os.environ.get(
    "VOXMM_CALLBACK_PATH",
    "callbacks/VoxMM/AV/EffConfInterCTC_finetune_visual_unfrozen_dev_eval",
)

# Greedy CTC, matching the other VoxMM runs.
beamsearch = False

# Training
batch_size = int(os.environ.get("VOXMM_BATCH_SIZE", "8"))
eval_batch_size = int(os.environ.get("VOXMM_EVAL_BATCH_SIZE", str(batch_size)))
accumulated_steps = int(os.environ.get("VOXMM_ACCUMULATED_STEPS", "8"))
eval_training = False
precision = torch.float32
recompute_metrics = True
epochs = int(os.environ.get("VOXMM_EPOCHS", "15"))
warmup_steps = int(os.environ.get("VOXMM_WARMUP_STEPS", "300"))
peak_lr = float(os.environ.get("VOXMM_PEAK_LR", "1e-4"))
min_lr = float(os.environ.get("VOXMM_MIN_LR", "1e-6"))
scheduler_steps = int(os.environ.get("VOXMM_SCHEDULER_STEPS", "9000"))

# Model
model = nnet.AudioVisualEfficientConformerInterCTC(
    vocab_size=vocab_size,
    v_interctc_blocks=v_interctc_blocks,
    a_interctc_blocks=a_interctc_blocks,
    f_interctc_blocks=f_interctc_blocks,
)
optimizer = nnet.Adam(
    params=model.parameters(),
    lr=nnet.CosineAnnealingScheduler(
        warmup_steps=warmup_steps,
        val_max=peak_lr,
        val_min=min_lr,
        end_step=scheduler_steps,
    ),
    betas=(0.9, 0.98),
    eps=1e-9,
    weight_decay=1e-6,
)
model.compile(
    losses=nnet.CTCLoss(zero_infinity=True, assert_shorter=False),
    decoders={"outputs": nnet.CTCGreedySearchDecoder(tokenizer_path=tokenizer_path)},
    metrics={"outputs": nnet.WordErrorRate()},
    loss_weights=loss_weights,
    optimizer=optimizer,
)

checkpoint = torch.load(pretrained_checkpoint, map_location=model.device)
model.load_state_dict(checkpoint["model_state_dict"], strict=True)
print("Loaded pretrained checkpoint:", pretrained_checkpoint)

# No visual freezing: video front-end, visual backend, audio path, and fusion stay trainable.
frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Frozen params:    {frozen:,} ({frozen/(frozen+trainable)*100:.1f}%)")
print(f"Trainable params: {trainable:,} ({trainable/(frozen+trainable)*100:.1f}%)")

# W&B logging
import wandb

wandb.init(
    project=os.environ.get("WANDB_PROJECT", "team1-asr"),
    name=os.environ.get("WANDB_RUN_NAME", "avec-voxmm-finetune-visual-unfrozen"),
    config={
        "model": "AudioVisualEfficientConformerInterCTC",
        "pretrained": pretrained_checkpoint,
        "frozen": [],
        "train_manifest": manifest_train,
        "eval_manifest": manifest_eval,
        "trainable_params": trainable,
        "frozen_params": frozen,
        "batch_size": batch_size,
        "accumulated_steps": accumulated_steps,
        "peak_lr": peak_lr,
        "min_lr": min_lr,
        "warmup_steps": warmup_steps,
        "scheduler_steps": scheduler_steps,
        "dataset": os.environ.get("VOXMM_DATASET_NAME", "VoxMM-train-no-dev-ids-dev-eval"),
    },
)

_original_log_step = model.log_step.__func__


def _wandb_log_step(self, losses, metrics, infos, writer, step, tag, dist_log=False):
    _original_log_step(self, losses, metrics, infos, writer, step, tag, dist_log)
    if self.rank == 0:
        log_dict = {f"{tag}/{key}": value for key, value in losses.items()}
        log_dict.update({f"{tag}/{key}": value for key, value in metrics.items()})
        log_dict.update({f"{tag}/{key}": float(value) for key, value in infos.items() if isinstance(value, (float, int))})
        wandb.log(log_dict, step=int(self.model_step))


import types
model.log_step = types.MethodType(_wandb_log_step, model)

# Dataset transforms
crop_size = (88, 88)
training_video_transform = nn.Sequential(
    torchvision.transforms.RandomCrop(crop_size),
    torchvision.transforms.RandomHorizontalFlip(),
    nnet.Permute(dims=(2, 3, 0, 1)),
    nnet.TimeMaskSecond(T_second=0.4, num_mask_second=1.0, fps=25.0, mean_frame=True),
    nnet.Permute(dims=(2, 3, 0, 1)),
)
evaluation_video_transform = torchvision.transforms.CenterCrop(crop_size)

align = True
video_max_length = 400
collate_fn = nnet.CollateFn(
    inputs_params=[{"axis": 0, "padding": True}, {"axis": 3}, {"axis": 1, "padding": True}, {"axis": 4}],
    targets_params=({"axis": 2, "padding": True}, {"axis": 5}),
)

training_dataset = nnet.datasets.VoxMM(
    batch_size=batch_size,
    collate_fn=collate_fn,
    manifest_path=manifest_train,
    tokenizer_path=tokenizer_path,
    video_max_length=video_max_length,
    video_transform=training_video_transform,
    align=align,
)

evaluation_dataset = nnet.datasets.VoxMM(
    batch_size=eval_batch_size,
    collate_fn=collate_fn,
    manifest_path=manifest_eval,
    tokenizer_path=tokenizer_path,
    video_transform=evaluation_video_transform,
    align=align,
)
