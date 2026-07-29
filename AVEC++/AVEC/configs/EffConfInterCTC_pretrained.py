import sys
sys.path.append("../")
sys.path.append("../SwinLip/SwinLip")  # github.com/park-ing/SwinLip clone -- ADJUST THIS

# Imports
import nnet
import torch
import torch.nn as nn
import torchvision

# Architecture
vocab_size = 256
v_interctc_blocks = [2]     # shallower than the from-scratch model -- SwinLip already
a_interctc_blocks = [2]     # does most of the representation learning; tune once you
f_interctc_blocks = [2]     # see training curves
# NOTE: model.compile() below defaults loss_weights to {"v_ctc_1", "a_ctc_1",
# "f_ctc_1", "outputs"} to match these block indices (loss_prefix + "_" + str(i),
# i is 0-indexed -- same math as your original config's v_ctc_2/v_ctc_5 etc., just
# shallower stacks here). Override loss_weights explicitly in compile() below if
# you change these block indices.

# SwinLip pretrained backbone
swinlip_config_path = "../SwinLip/SwinLip/configs/swinlip.json"
swinlip_checkpoint_path = "../SwinLip/lrw_swinlip.pth"

# wav2vec2 pretrained backbone
wav2vec2_name_or_path = "facebook/wav2vec2-base-960h"

# Freeze pretrained backbones, train everything else
freeze_swinlip = True
freeze_wav2vec2 = True

# Training
batch_size = 16
accumulated_steps = 4
eval_training = False
precision = torch.float16
recompute_metrics = True
callback_path = "callbacks/LRS23/AV/EffConfInterCTC_Pretrained"

# Beam Search (unchanged from your original config)
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
    freeze_wav2vec2=freeze_wav2vec2,
)
model.compile(
    losses=nnet.CTCLoss(zero_infinity=True, assert_shorter=False),
    decoders={"outputs": nnet.CTCGreedySearchDecoder(tokenizer_path=tokenizer_path) if not beamsearch else nnet.CTCBeamSearchDecoder(tokenizer_path=tokenizer_path, beam_size=beam_size, ngram_path=ngram_path, ngram_tmp=ngram_tmp, ngram_alpha=ngram_alpha, ngram_beta=ngram_beta, ngram_offset=ngram_offset, neural_config_path=neural_config_path, neural_checkpoint=neural_checkpoint, neural_alpha=neural_alpha, neural_beta=neural_beta)},
    metrics={"outputs": nnet.WordErrorRate()},
)
# NOTE: loss_weights intentionally omitted above -- AudioVisualPretrainedConformerInterCTC.compile()
# already defaults to the right keys (v_ctc_1/a_ctc_1/f_ctc_1/outputs) for the
# v_interctc_blocks=[2] etc. set above. Override explicitly here if you change
# those block indices.

# NOTE: the old `lrw_pretrained` block that surgically loaded a state dict into
# `model.encoder.video_encoder.front_end` is REMOVED here -- SwinLipFeatureExtractor
# already loads its checkpoint inside its own __init__ (called internally by
# AudioVisualPretrainedConformerInterCTC now), so there's nothing left to do at
# this point in the config.

# Datasets (unchanged from your original config)
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
training_dataset = nnet.datasets.MultiDataset(
    batch_size=batch_size,
    collate_fn=collate_fn,
    datasets=[
        nnet.datasets.LRS(batch_size=None, collate_fn=None, version="LRS3", mode="train", video_max_length=video_max_length, video_transform=training_video_transform, align=align),
        # LRS3 has no data under datasets/LRS3 yet (manifest-built stand-in only covers LRS2 so far) --
        # a 0-sample LRS3 entry here is harmless for training (MultiDataset just sums lengths), but the
        # equivalent 0-sample entry below in evaluation_dataset crashes DataLoader's RandomSampler.
        # Re-add once datasets/LRS3 is populated (e.g. via tools/build_lrs2_from_manifest.py --version LRS3).
        # nnet.datasets.LRS(batch_size=None, collate_fn=None, version="LRS3", mode="pretrain+trainval", video_max_length=video_max_length, video_transform=training_video_transform, align=align)
    ])

evaluation_dataset = [
    nnet.datasets.LRS(batch_size=batch_size, collate_fn=collate_fn, version="LRS3", mode="test", video_transform=evaluation_video_transform, align=align),
    # See note above -- re-add once datasets/LRS3 has data.
    # nnet.datasets.LRS(batch_size=batch_size, collate_fn=collate_fn, version="LRS3", mode="test", video_transform=evaluation_video_transform, align=align)
]