import sys
sys.path.append("../../")

import nnet
import torch
import torch.nn as nn
import torchvision

# Architecture (same as pretrained model)
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
    "outputs": 0.5
}

# Paths
tokenizer_path = "datasets/LRS3/tokenizerbpe256.model"
pretrained_checkpoint = "callbacks/LRS23/AV/EffConfInterCTC/checkpoints_swa-equal-60-70.ckpt"
manifest_train = "/hpcwork/ww202463/voxmm_preprocessed/manifest.csv"
manifest_eval = "/hpcwork/ww202463/team-1-data/lrs_voxmm/manifest_dev_eval_avec_roi/manifest.csv"
callback_path = "callbacks/VoxMM/AV/EffConfInterCTC_finetune"

# Beam Search - disabled (no ctcdecode)
beamsearch = False

# Training
batch_size = 8
accumulated_steps = 8  # effective batch = 64
eval_training = False
precision = torch.float32
recompute_metrics = True
epochs = 15

# Optimizer defined after model (needs model.parameters())

# Model
model = nnet.AudioVisualEfficientConformerInterCTC(
    vocab_size=vocab_size,
    v_interctc_blocks=v_interctc_blocks,
    a_interctc_blocks=a_interctc_blocks,
    f_interctc_blocks=f_interctc_blocks,
)
# 38,854 samples / (8 batch * 8 accum) = ~607 optimizer steps/epoch
# 15 epochs * 607 steps = ~9,100 total steps
optimizer = nnet.Adam(
    params=model.parameters(),
    lr=nnet.CosineAnnealingScheduler(
        warmup_steps=300,
        val_max=1e-4,
        val_min=1e-6,
        end_step=9000,
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

# Load pretrained AV checkpoint
checkpoint = torch.load(pretrained_checkpoint, map_location=model.device)
model.load_state_dict(checkpoint["model_state_dict"], strict=True)
print("Loaded pretrained checkpoint:", pretrained_checkpoint)

# Freeze visual front-end (ResNet-18+3DConv) and visual back-end stage 2
for param in model.encoder.video_encoder.front_end.parameters():
    param.requires_grad = False
for param in model.encoder.video_encoder.back_end.conformer_blocks[6].parameters():
    param.requires_grad = False

frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Frozen params:    {frozen:,} ({frozen/(frozen+trainable)*100:.1f}%)")
print(f"Trainable params: {trainable:,} ({trainable/(frozen+trainable)*100:.1f}%)")

# W&B logging
import wandb
import os

wandb.init(
    project="team1-asr",
    name="avec-voxmm-finetune-full",
    config={
        "model": "AudioVisualEfficientConformerInterCTC",
        "pretrained": pretrained_checkpoint,
        "frozen": ["video_encoder.front_end", "video_encoder.back_end.block6"],
        "trainable_params": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "frozen_params": sum(p.numel() for p in model.parameters() if not p.requires_grad),
        "batch_size": batch_size,
        "accumulated_steps": accumulated_steps,
        "peak_lr": 1e-4,
        "warmup_steps": 50,
        "dataset": "VoxMM-smoke",
    }
)

# Monkey-patch log_step to also log to W&B
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
    batch_size=batch_size,
    collate_fn=collate_fn,
    manifest_path=manifest_eval,
    tokenizer_path=tokenizer_path,
    video_transform=evaluation_video_transform,
    align=align,
)
