import sys
sys.path.append("../")
sys.path.append("../SwinLip/SwinLip")  # github.com/park-ing/SwinLip clone -- ADJUST THIS

# Imports
import nnet
import torch
import torch.nn as nn
import torchvision

# CLAIX CONFIG
# Same architecture as EffConfInterCTC_pretrained.py. Only two things changed
# for the cluster:
#   1) wav2vec2_name_or_path points at the localized checkpoint (../wav2vec2),
#      same as EffConfInterCTC_pretrained_smoke.py -- avoids a HF Hub hit,
#      which matters since CLAIX compute nodes have no internet.
#   2) The dataset block below loads the CLAIX copy of the data. Per Alexis
#      (2026-07-26): that data is LRS3 content but packaged LRS2-shaped
#      (root/LRS2/mvlrs_v1/{pretrain,main}/<speaker>/<clip>_mouth.mp4 + .flac +
#      .pt, plus root/LRS2/{train,val,test,pretrain}.txt filelists) -- exactly
#      like datasets_smoke/LRS2 was built via tools/build_lrs2_from_manifest.py.
#      So this loads with version="LRS2", NOT version="LRS3", same reasoning
#      as the NOTE at the top of configs/LRS23/AV/EffConfInterCTC_pretrained_smoke.py.
#      Do not copy version="LRS3" from the original EffConfInterCTC_pretrained.py.

# Architecture (unchanged)
vocab_size = 256
v_interctc_blocks = [2]
a_interctc_blocks = [2]
f_interctc_blocks = [2]

# SwinLip pretrained backbone
swinlip_config_path = "../SwinLip/SwinLip/configs/swinlip.json"
swinlip_checkpoint_path = "../SwinLip/lrw_swinlip.pth"

# wav2vec2 pretrained backbone -- localized checkpoint, no HF Hub download needed
wav2vec2_name_or_path = "../wav2vec2/wav2vec2-base-960h"

# Freeze pretrained backbones, train everything else
freeze_swinlip = True
freeze_wav2vec2 = True

# Partial unfreezing (added 2026-07-28, after joint_prob=0.5->0.8): hard-eval WER
# (Evaluation-epoch/1/wer) sat flat around 80% for its first 4 measured epochs while
# frozen -- consistent with the frozen wav2vec2 features themselves being a ceiling,
# since HardArtificialNoise only distorts the audio channel and all adaptation
# capacity currently sits downstream of a fixed feature extractor that never saw
# reverb/heavy-SNR-noise/bandwidth-limiting in its own pretraining. Unfreezing the
# last 2 wav2vec2 transformer layers lets them adapt directly instead.
# SwinLip's module tree (inspected 2026-07-28 via tools/inspect_swinlip_modules.py):
# model.trunk = SwinTransformer(patch_embed -> layers.{0,1,2} (Swin stages,
# image-level attention/MLP, deepest/heaviest part) -> norm -> avgpool ->
# CM_encoder (SwinLip's own trailing 2-layer Conformer that feeds this repo's
# adapter directly)). CM_encoder.encoder.layers.{0,1} is the closest analogue to
# wav2vec2's last-N-transformer-layers: the last processing stage before our own
# conformer picks it up, not the deep frozen Swin image backbone underneath it.
# Left EMPTY (not wired in) for now, though, since HardArtificialNoise only
# distorts the audio channel -- unfreezing wav2vec2 is the higher-leverage lever
# for the hard-noise-robustness objective specifically. Revisit this once the
# wav2vec2 unfreezing result is in; if/when you do, note CM_encoder only has 2
# layers total, so listing both is unfreezing the whole trailing conformer, not a
# small slice of it (the Swin stages above it stay frozen either way):
# swinlip_unfreeze_modules = ["model.trunk.CM_encoder.encoder.layers.0", "model.trunk.CM_encoder.encoder.layers.1"]
wav2vec2_unfreeze_last_n_layers = 2
swinlip_unfreeze_modules = []

# Training
# TODO: batch_size/accumulated_steps were sized for a single dev GPU with the
# 8-sample smoke set. Re-check against the real dataset size and whatever GPU
# allocation your sbatch script requests (--gres=gpu:N) before the real run.
batch_size = 16
accumulated_steps = 4
# epochs = 1000          # main.py default if unset -- set an explicit number once
                          # you've seen a few epochs of real training curves.
eval_training = False
precision = torch.float16
recompute_metrics = True
callback_path = "callbacks/LRS3_CLAIX/EffConfInterCTC_Pretrained"

# Beam Search (off for training -- unchanged)
beamsearch = False
beam_size = 16
tokenizer_path = "../tokenizer/tokenizerbpe256.model"
ngram_path = "datasets/LRS3/6gram_lrs23.arpa"
ngram_offset = 100
ngram_alpha = 0.6
ngram_beta = 1.0
ngram_tmp = 1.0
neural_config_path = "configs/LRS23/LM/GPT-Small.py"
neural_checkpoint = "checkpoints_epoch_10_step_2860.ckpt"
neural_alpha = 0.6
neural_beta = 1.0

# Model
model = nnet.AudioVisualPretrainedConformerInterCTC(
    vocab_size=vocab_size,
    v_interctc_blocks=v_interctc_blocks,
    a_interctc_blocks=a_interctc_blocks,
    f_interctc_blocks=f_interctc_blocks,
    swinlip_config_path=swinlip_config_path,
    swinlip_checkpoint_path=swinlip_checkpoint_path,
    wav2vec2_name_or_path=wav2vec2_name_or_path,
    freeze_swinlip=freeze_swinlip,
    swinlip_unfreeze_modules=swinlip_unfreeze_modules,
    freeze_wav2vec2=freeze_wav2vec2,
    wav2vec2_unfreeze_last_n_layers=wav2vec2_unfreeze_last_n_layers,
)
model.compile(
    losses=nnet.CTCLoss(zero_infinity=True, assert_shorter=False),
    decoders={"outputs": nnet.CTCGreedySearchDecoder(tokenizer_path=tokenizer_path) if not beamsearch else nnet.CTCBeamSearchDecoder(tokenizer_path=tokenizer_path, beam_size=beam_size, ngram_path=ngram_path, ngram_tmp=ngram_tmp, ngram_alpha=ngram_alpha, ngram_beta=ngram_beta, ngram_offset=ngram_offset, neural_config_path=neural_config_path, neural_checkpoint=neural_checkpoint, neural_alpha=neural_alpha, neural_beta=neural_beta)},
    metrics={"outputs": nnet.WordErrorRate()},
    # Only takes effect when wav2vec2_unfreeze_last_n_layers>0 / swinlip_unfreeze_modules
    # is non-empty (see AudioVisualPretrainedConformerInterCTC.compile() in
    # nnet/models_zoo.py) -- the main Noam schedule (warmup_steps=10000) for
    # everything else is untouched and keeps decaying exactly as it has been.
    # This gives the newly-unfrozen wav2vec2 layers their OWN short warmup +
    # low-peak schedule since they're entering the optimizer for the first time,
    # rather than either restarting the main schedule's warmup (would disrupt
    # ~60+ epochs of already-converged decay for no reason) or bolting them onto
    # it at whatever LR the main schedule happens to be at right now (tuned for a
    # from-scratch stack, not for fine-tuning pretrained weights).
    encoder_finetune_warmup_steps=1000,
    encoder_finetune_val_factor=0.1,
)
# loss_weights intentionally omitted -- compile() default {"v_ctc_1", "a_ctc_1",
# "f_ctc_1", "outputs"} already matches v_/a_/f_interctc_blocks=[2] above.

# Datasets
align = True
video_max_length = 400
collate_fn = nnet.CollateFn(inputs_params=[{"axis": 0, "padding": True}, {"axis": 3}, {"axis": 1, "padding": True}, {"axis": 4}], targets_params=({"axis": 2, "padding": True}, {"axis": 5}))
crop_size = (88, 88)
training_video_transform = nn.Sequential(
    torchvision.transforms.RandomCrop(crop_size),
    torchvision.transforms.RandomHorizontalFlip(),
    nnet.Permute(dims=(2, 3, 0, 1)),
    nnet.TimeMaskSecond(T_second=0.4, num_mask_second=1.0, fps=25.0, mean_frame=True),
    nnet.Permute(dims=(2, 3, 0, 1))
)
evaluation_video_transform = torchvision.transforms.CenterCrop(crop_size)

# Hard artificial noise augmentation (reverb -> DEMAND -> bandwidth-limiting filter),
# reproducing team-1's LRS-VoxMM-style degradation recipe (see AVEC++/noise.py) but applied
# on-the-fly and in-memory during training instead of writing a fixed noisy copy of the
# dataset to disk. Runs fresh every time LRS.__getitem__ is called, i.e. every epoch --
# no extra epoch-tracking needed, see nnet/transforms.py's HardArtificialNoise docstring.
# Training-only: evaluation_dataset below intentionally gets no audio_transform, so
# validation WER still tracks clean LRS2/3.
# Reverb is intentionally left ON -- it's a big part of what makes LRS-VoxMM-style noise
# hard, don't set it to 0 to work around a missing pyroomacoustics install; fix the
# install instead (CLAIX_SETUP.md step 2).
# ADJUST THIS: must match wherever you rsync'd DEMAND to on CLAIX -- see
# CLAIX_SETUP.md step 3c. Config import fails immediately if this path doesn't exist.
demand_noise_path = "/rwthfs/rz/cluster/hpcwork/lect0176/team-1/data/DEMAND/16k"

# Parameters below are LRS-VoxMM's published "3-dist hard" synthetic distortion split,
# spelled out explicitly here rather than relying on HardArtificialNoise's defaults (which
# match this exactly, but keeping the numbers visible in the config makes this config
# self-documenting against the paper without cross-referencing nnet/transforms.py):
#   - reverb: room length/width in [5, 15] m, height in [2, 6] m, RT60 in [0.4, 1.0] s
#   - DEMAND additive noise: SNR in [-5, 0] dB ("hard" -- their "easy" split is [5, 15] dB)
#   - bandwidth: butter/bessel/cheby1 low-pass, cutoff in {2, 4, 8} kHz
# filter_order isn't specified in their writeup; left at a reasonable default (4).
#
# joint_prob (not per-stage probs) mixes clean and hard-distorted samples: with this
# probability all three stages fire together (an exact "3-dist hard" sample), otherwise the
# sample is left clean -- some clean exposure for training stability, without diluting the
# other half into partial/non-standard distortion combinations. Started at 0.5 given the
# model was still very early ("et vet vet" collapse) when this was added; raised to 0.8 once
# training was well past that (per Alexis, 2026-07-28) to spend more training on the real
# target condition -- clean WER may take a modest hit from the reduced clean exposure, that's
# an expected tradeoff, not a bug (see Evaluation-epoch/0/wer alongside Evaluation-epoch/1/wer).
training_audio_transform = nnet.HardArtificialNoise(
    demand_noise_path=demand_noise_path,
    joint_prob=0.8,
    reverb_length_range=(5.0, 15.0),
    reverb_width_range=(5.0, 15.0),
    reverb_height_range=(2.0, 6.0),
    reverb_rt60_range=(0.4, 1.0),
    demand_snr_db_range=(-5.0, 0.0),
    bandwidth_filters=("butter", "bessel", "cheby1"),
    bandwidth_cutoffs_hz=(2000, 4000, 8000),
    bandwidth_filter_order=4,
)

# Second copy of the same recipe, but joint_prob=1.0 instead of 0.5 -- for evaluation we
# want every sample to get the full "3-dist hard" treatment (deterministically applied,
# not the 50/50 clean/hard mix used for training), so this tracks the actual target
# condition directly instead of only being measured indirectly through training exposure.
# Room dims / SNR / filter choice still randomize per sample (matches how the paper's own
# benchmark was built), only whether all three stages fire is no longer a coin flip.
evaluation_hard_audio_transform = nnet.HardArtificialNoise(
    demand_noise_path=demand_noise_path,
    joint_prob=1.0,
    reverb_length_range=(5.0, 15.0),
    reverb_width_range=(5.0, 15.0),
    reverb_height_range=(2.0, 6.0),
    reverb_rt60_range=(0.4, 1.0),
    demand_snr_db_range=(-5.0, 0.0),
    bandwidth_filters=("butter", "bessel", "cheby1"),
    bandwidth_cutoffs_hz=(2000, 4000, 8000),
    bandwidth_filter_order=4,
)

# Built via tools/build_lrs2_from_manifest.py from the raw LRS3 trainval
# manifest (/rwthfs/rz/cluster/hpcwork/lect0176/team-1/data/lrs3/avec_roi/manifest.csv),
# which we split 80/20 into train/val (tools/split_manifest_train_val.py) since
# that manifest has no LRS3 test-split data in it. Lands at the tool's default
# --root=datasets, i.e. relative to cwd -- safe here because main.py, the
# sbatch script, and the smoke test script all enforce running from inside
# AVEC++/AVEC/ (they check for main.py's presence before doing anything else).
data_root = "datasets"

training_dataset = nnet.datasets.MultiDataset(
    batch_size=batch_size,
    collate_fn=collate_fn,
    datasets=[
        nnet.datasets.LRS(root=data_root, batch_size=None, collate_fn=None, version="LRS2", mode="train", video_max_length=video_max_length, video_transform=training_video_transform, audio_transform=training_audio_transform, align=align),
    ])

evaluation_dataset = [
    # mode="val", not "test" -- the manifest we built from is LRS3 trainval only.
    # index 0: clean val -- this is what's been logged as "Evaluation-epoch/0/*" so far.
    nnet.datasets.LRS(root=data_root, batch_size=batch_size, collate_fn=collate_fn, version="LRS2", mode="val", video_transform=evaluation_video_transform, align=align),
    # index 1: same val split, full "3-dist hard" applied to every sample -- logs as
    # "Evaluation-epoch/1/*". This is the number that actually reflects LRS-VoxMM-hard-style
    # robustness; index 0 alone can't tell you that since it's always clean by design.
    nnet.datasets.LRS(root=data_root, batch_size=batch_size, collate_fn=collate_fn, version="LRS2", mode="val", video_transform=evaluation_video_transform, audio_transform=evaluation_hard_audio_transform, align=align),
]
