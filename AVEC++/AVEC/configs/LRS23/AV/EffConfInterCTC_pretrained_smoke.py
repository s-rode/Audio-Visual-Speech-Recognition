import sys
sys.path.append("../")
sys.path.append("../SwinLip/SwinLip")  # github.com/park-ing/SwinLip clone -- ADJUST THIS

# Imports
import nnet
import torch
import torch.nn as nn
import torchvision

# SMOKE TEST CONFIG
# 8-utterance slice of LRS3 (test_lrs3/manifest/test_manifest.csv), built into
# datasets_smoke/LRS2/... via tools/build_lrs2_from_manifest.py (run once with
# --split train, once with --split test -- both point at the same 8 clips).
# NOTE: the tool has no --version flag, it only writes LRS2-shaped output, so
# this config loads it back with version="LRS2" even though the underlying
# clips are LRS3. Do not copy version="LRS3" from the real training config here.

# Architecture (unchanged from EffConfInterCTC_pretrained.py)
vocab_size = 256
v_interctc_blocks = [2]
a_interctc_blocks = [2]
f_interctc_blocks = [2]

# SwinLip pretrained backbone
swinlip_config_path = "../SwinLip/SwinLip/configs/swinlip.json"
swinlip_checkpoint_path = "../SwinLip/lrw_swinlip.pth"

# wav2vec2 pretrained backbone
wav2vec2_name_or_path = "facebook/wav2vec2-base-960h"

# Freeze pretrained backbones, train everything else
freeze_swinlip = True
freeze_wav2vec2 = True

# Training -- shrunk to fit an 8-sample dataset
batch_size = 4              # must divide cleanly into 8 (drop_last=True on the train loader)
accumulated_steps = 1        # no need to accumulate over a dataset this small
epochs = 2                   # main.py defaults to 1000 epochs if unset -- don't let this run away
eval_training = False
precision = torch.float16    # requires CUDA autocast -- switch to torch.float32 + pass --cpu if testing without a GPU
recompute_metrics = True
callback_path = "callbacks/smoke_test/EffConfInterCTC_Pretrained"  # isolated from real training callbacks

# Beam Search (unchanged -- stays off, ngram/neural LM files aren't needed for a smoke test)
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
# loss_weights intentionally omitted -- compile() default {"v_ctc_1", "a_ctc_1", "f_ctc_1", "outputs"}
# already matches v_/a_/f_interctc_blocks=[2] above.

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

# root="datasets_smoke" + version="LRS2" -- see NOTE at top of file
training_dataset = nnet.datasets.MultiDataset(
    batch_size=batch_size,
    collate_fn=collate_fn,
    datasets=[
        nnet.datasets.LRS(batch_size=None, collate_fn=None, root="datasets_smoke", version="LRS2", mode="train", video_max_length=video_max_length, video_transform=training_video_transform, align=align),
    ])

evaluation_dataset = [
    nnet.datasets.LRS(batch_size=batch_size, collate_fn=collate_fn, root="datasets_smoke", version="LRS2", mode="test", video_transform=evaluation_video_transform, align=align),
]
