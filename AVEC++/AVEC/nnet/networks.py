# Copyright 2021, Maxime Burchi. Modified to add pretrained SwinLip / wav2vec2 front-ends.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# PyTorch
import json
import torch
import torch.nn as nn
import torch.nn.functional as F

# NeuralNets
from nnet import preprocessing
from nnet import layers
from nnet import modules
from nnet import blocks
from nnet import attentions
from nnet import transforms
from nnet import normalizations

# wav2vec 2.0 (HuggingFace) -- used by AudioWav2Vec2ConformerEncoder below
try:
    from transformers import Wav2Vec2Model
except ImportError:
    Wav2Vec2Model = None

# SwinLip (https://github.com/park-ing/SwinLip) -- clone it and add it to your
# PYTHONPATH, e.g.: import sys; sys.path.append("/path/to/SwinLip")
# before importing nnet, or install it as an editable local package.
# Used by SwinLipFeatureExtractor below.
try:
    from models.swinlip import SwinLip, threeD_to_2D_tensor
except ImportError:
    SwinLip = None
    threeD_to_2D_tensor = None

###############################################################################
# Networks
###############################################################################

class ResNet(nn.Module):

    """ ResNet (ResNet18, ResNet34, ResNet50, ResNet101, ResNet152)

    Models: 224 x 224
    ResNet18: 11,689,512 Params
    ResNet34: 21,797,672 Params
    ResNet50: 25,557,032 Params
    ResNet101: 44,549,160 Params
    Resnet152: 60,192,808 Params

    Reference: "Deep Residual Learning for Image Recognition" by He et al.
    https://arxiv.org/abs/1512.03385

    """

    def __init__(self, dim_input=3, dim_output=1000, model="ResNet50", include_stem=True, include_head=True):
        super(ResNet, self).__init__()

        assert model in ["ResNet18", "ResNet34", "ResNet50", "ResNet101", "ResNet152"]

        if model == "ResNet18":
            dim_stem = 64
            dim_blocks = [64, 128, 256, 512]
            num_blocks = [2, 2, 2, 2]
            bottleneck = False
        elif model == "ResNet34":
            dim_stem = 64
            dim_blocks = [64, 128, 256, 512]
            num_blocks = [3, 4, 6, 3]
            bottleneck = False
        elif model == "ResNet50":
            dim_stem = 64
            dim_blocks = [256, 512, 1024, 2048]
            num_blocks = [3, 4, 6, 3]
            bottleneck = True
        elif model == "ResNet101":
            dim_stem = 64
            dim_blocks = [256, 512, 1024, 2048]
            num_blocks = [3, 4, 23, 3]
            bottleneck = True
        elif model == "ResNet152":
            dim_stem = 64
            dim_blocks = [256, 512, 1024, 2048]
            num_blocks = [3, 8, 36, 3]
            bottleneck = True

        self.stem = nn.Sequential(
            layers.Conv2d(in_channels=dim_input, out_channels=dim_stem, kernel_size=(7, 7), stride=(2, 2), weight_init="he_normal", bias=False),
            normalizations.BatchNorm2d(num_features=dim_stem),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        ) if include_stem else nn.Identity()

        # Blocks
        self.blocks = nn.ModuleList()
        for stage_id in range(4):

            for block_id in range(num_blocks[stage_id]):

                # Projection Block
                if block_id == 0:
                    if stage_id == 0:
                        stride = (1, 1)
                        bottleneck_ratio = 1
                        in_features = dim_stem
                    else:
                        stride = (2, 2)
                        bottleneck_ratio = 2
                        in_features = dim_blocks[stage_id-1]
                # Default Block
                else:
                    stride = (1, 1)
                    in_features = dim_blocks[stage_id]
                    bottleneck_ratio = 4

                if bottleneck:
                    self.blocks.append(blocks.ResNetBottleneckBlock(
                        in_features=in_features,
                        out_features=dim_blocks[stage_id],
                        bottleneck_ratio=bottleneck_ratio,
                        kernel_size=(3, 3),
                        stride=stride,
                        act_fun="ReLU",
                        joined_post_act=True
                    ))
                else:
                    self.blocks.append(blocks.ResNetBlock(
                        in_features=in_features,
                        out_features=dim_blocks[stage_id],
                        kernel_size=(3, 3),
                        stride=stride,
                        act_fun="ReLU",
                        joined_post_act=True
                    ))

        # Head
        self.head = nn.Sequential(
            layers.GlobalAvgPool2d(),
            layers.Linear(in_features=dim_blocks[-1], out_features=dim_output, weight_init="he_normal", bias_init="zeros")
        ) if include_head else nn.Identity()

    def forward(self, x):

        # (B, Din, H, W) -> (B, D0, H//4, W//4)
        x = self.stem(x)

        # (B, D0, H//4, W//4) -> (B, D4, H//32, W//32)
        for block in self.blocks:
            x = block(x)

        # (B, D4, H//32, W//32) -> (B, Dout)
        x = self.head(x)

        return x

class Transformer(nn.Module):

    def __init__(self, dim_model, num_blocks, att_params={"class": "MultiHeadAttention", "params":{"num_heads": 4, "weight_init": "normal_02", "bias_init": "zeros"}}, ff_ratio=4, emb_drop_rate=0.1, drop_rate=0.1, act_fun="GELU", pos_embedding=None, mask=None, inner_dropout=False, weight_init="normal_02", bias_init="zeros", post_norm=False):
        super(Transformer, self).__init__()

        # Positional Embedding
        self.pos_embedding = pos_embedding

        # Input Dropout
        self.dropout = nn.Dropout(p=emb_drop_rate)

        # Mask
        self.mask = mask

        # Transformer Blocks
        self.blocks = nn.ModuleList([blocks.TransformerBlock(
            dim_model=dim_model,
            ff_ratio=ff_ratio,
            att_params=att_params,
            drop_rate=drop_rate,
            inner_dropout=inner_dropout,
            act_fun=act_fun,
            weight_init=weight_init,
            bias_init=bias_init,
            post_norm=post_norm
        ) for block_id in range(num_blocks)])

        # LayerNorm
        self.layernorm = nn.LayerNorm(normalized_shape=dim_model) if not post_norm else nn.Identity()

    def forward(self, x, lengths=None):

        # Pos Embedding
        if self.pos_embedding != None:
            x = self.pos_embedding(x)

        # Input Dropout
        x = self.dropout(x)

        # Mask (1 or B, 1, N, N)
        if self.mask != None:
            mask = self.mask(x, lengths)
        else:
            mask = None

        # Transformer Blocks
        for block in self.blocks:
            x = block(x, mask=mask)

        # LayerNorm
        x = self.layernorm(x)

        return x

class ConformerInterCTC(nn.Module):

    def __init__(self, dim_model, num_blocks, interctc_blocks, vocab_size, loss_prefix="ctc", att_params={"class": "MultiHeadAttention", "num_heads": 4}, conv_params={"class": "Conv1d", "params": {"padding": "same", "kernel_size": 31}}, ff_ratio=4, drop_rate=0.1, pos_embedding=None, mask=None, conv_stride=1, batch_norm=True):
        super(ConformerInterCTC, self).__init__()

        # Inter CTC Params
        self.interctc_blocks = interctc_blocks
        self.loss_prefix = loss_prefix

        # Single Stage
        if isinstance(dim_model, int):
            dim_model = [dim_model]
        if isinstance(num_blocks, int):
            num_blocks = [num_blocks]
        
        # Positional Embedding
        self.pos_embedding = pos_embedding

        # Input Dropout
        self.dropout = nn.Dropout(p=drop_rate)

        # Mask
        self.mask = mask

        # Conformer Stages
        i = 1
        self.conformer_blocks = nn.ModuleList()
        self.interctc_modules = nn.ModuleList()
        for stage_id in range(len(num_blocks)):

            # Conformer Blocks
            for block_id in range(num_blocks[stage_id]):

                # Transposed Block
                transposed_block = "Transpose" in conv_params["class"]

                # Downsampling Block
                down_block = ((block_id == 0) and (stage_id > 0)) if transposed_block else ((block_id == num_blocks[stage_id] - 1) and (stage_id < len(num_blocks) - 1))

                # Block
                self.conformer_blocks.append(blocks.ConformerBlock(
                    dim_model=dim_model[stage_id - (1 if transposed_block and down_block else 0)],
                    dim_expand=dim_model[stage_id + (1 if not transposed_block and down_block else 0)],
                    ff_ratio=ff_ratio,
                    drop_rate=drop_rate,
                    att_params=att_params[stage_id - (1 if transposed_block and down_block else 0)] if isinstance(att_params, list) else att_params,
                    conv_stride=1 if not down_block else conv_stride[stage_id] if isinstance(conv_stride, list) else conv_stride,
                    conv_params=conv_params[stage_id] if isinstance(conv_params, list) else conv_params,
                    batch_norm=batch_norm
                ))

                # InterCTC Block
                if i in interctc_blocks:
                    self.interctc_modules.append(modules.InterCTCResModule(
                        dim_model=dim_model[stage_id + (1 if not transposed_block and down_block else 0)], 
                        vocab_size=vocab_size
                    ))

                i += 1

    def forward(self, x, lengths):

        # Pos Embedding
        if self.pos_embedding != None:
            x = self.pos_embedding(x)
            
        # Dropout
        x = self.dropout(x)

        # Mask (1 or B, 1, N, N)
        if self.mask != None:
            mask = self.mask(x, lengths)
        else:
            mask = None

        # Conformer Blocks
        interctc_outputs = {}
        j = 0
        for i, block in enumerate(self.conformer_blocks):

            # Conformer Block
            x = block(x, mask=mask)

            # InterCTC Block
            if i + 1 in self.interctc_blocks:
                x, logits = self.interctc_modules[j](x)
                j += 1
                key = self.loss_prefix + "_" + str(i)
            else:
                logits = None

            # Strided Block
            if block.stride > 1:

                # Stride Mask (1 or B, 1, T // S, T // S)
                if mask is not None:
                    mask = mask[:, :, ::block.stride, ::block.stride]

                # Update Seq Lengths
                if lengths is not None:
                    lengths = torch.div(lengths - 1, block.stride, rounding_mode='floor') + 1

            if logits != None:
                interctc_outputs[key] = [logits, lengths]

        return x, lengths, interctc_outputs

class AudioEfficientConformerEncoder(nn.Module):

    def __init__(self, include_head=True, vocab_size=256, att_type="patch", interctc_blocks=[3, 6, 10, 13], num_blocks=[5, 6, 5], loss_prefix="ctc"):
        super(AudioEfficientConformerEncoder, self).__init__()

        assert att_type in ["regular", "grouped", "patch"]

        # Params
        sample_rate=16000
        n_fft=512
        win_length_ms=25
        hop_length_ms=10
        n_mels=80
        mF=2
        F=27
        mT=5
        pS=0.05
        kernel_size=15
        drop_rate=0.1
        attn_drop_rate=0.0
        max_pos_encoding=10000
        causal=False
        subsampling_filters=180
        dim_model=[180, 256, 360]
        num_heads=4

        # Audio Preprocessing (B, T) -> (B, n_mels, T // hop_length + 1)
        self.audio_preprocessing = preprocessing.AudioPreprocessing(
            sample_rate=sample_rate,
            n_fft=n_fft,
            win_length_ms=win_length_ms,
            hop_length_ms=hop_length_ms,
            n_mels=n_mels,
            normalize=False,
            mean=-5.6501,
            std=4.2280
        )

        # Spec Augment
        self.spec_augment = preprocessing.SpecAugment(
            mF=mF,
            F=F,
            mT=mT,
            pS=pS
        )

        # Unsqueeze (B, N, T) -> (B, 1, N, T)
        self.unsqueeze = layers.Unsqueeze(dim=1)

        # Stem (B, 1, N, T) -> (B, C, N', T')
        self.subsampling_module = modules.ConvNeuralNetwork(
            dim_input=1,
            dim_layers=subsampling_filters,
            kernel_size=3,
            strides=2,
            norm="BatchNorm2d",
            act_fun="Swish",
            drop_rate=0.0,
            dim=2
        )

        # Reshape (B, C, N, T) -> (B, D, T)
        self.reshape = layers.Reshape(shape=(subsampling_filters * n_mels // 2, -1), include_batch=False)

        # Transpose (B, D, T) -> (B, T, D)
        self.transpose = layers.Transpose(1, 2)

        # Linear Proj
        self.linear = layers.Linear(subsampling_filters * n_mels // 2, dim_model[0])

        # Conformer
        self.back_end = ConformerInterCTC(
            dim_model=dim_model,
            num_blocks=num_blocks,
            interctc_blocks=interctc_blocks,
            vocab_size=vocab_size,
            att_params=[
                {"class": "RelPos1dMultiHeadAttention", "params": {"num_heads": num_heads, "attn_drop_rate": attn_drop_rate, "num_pos_embeddings": max_pos_encoding, "weight_init": "default", "bias_init": "default"}},
                {"class": "RelPos1dMultiHeadAttention", "params": {"num_heads": num_heads, "attn_drop_rate": attn_drop_rate, "num_pos_embeddings": max_pos_encoding, "weight_init": "default", "bias_init": "default"}},
                {"class": "RelPos1dMultiHeadAttention", "params": {"num_heads": num_heads, "attn_drop_rate": attn_drop_rate, "num_pos_embeddings": max_pos_encoding, "weight_init": "default", "bias_init": "default"}}
            ] if att_type == "regular" else [
                {"class": "GroupedRelPosMultiHeadSelfAttention", "params": {"num_heads": num_heads, "group_size": 3, "attn_drop_rate": attn_drop_rate, "max_pos_encoding": max_pos_encoding, "causal": causal}},
                {"class": "GroupedRelPosMultiHeadSelfAttention", "params": {"num_heads": num_heads, "group_size": 1, "attn_drop_rate": attn_drop_rate, "max_pos_encoding": max_pos_encoding, "causal": causal}},
                {"class": "GroupedRelPosMultiHeadSelfAttention", "params": {"num_heads": num_heads, "group_size": 1, "attn_drop_rate": attn_drop_rate, "max_pos_encoding": max_pos_encoding, "causal": causal}}
            ] if att_type == "grouped" else [
                {"class": "RelPosPatch1dMultiHeadAttention", "params": {"num_heads": num_heads, "patch_size": 3, "attn_drop_rate": attn_drop_rate, "num_pos_embeddings": max_pos_encoding, "weight_init": "default", "bias_init": "default"}},
                {"class": "RelPos1dMultiHeadAttention", "params": {"num_heads": num_heads, "attn_drop_rate": attn_drop_rate, "num_pos_embeddings": max_pos_encoding, "weight_init": "default", "bias_init": "default"}},
                {"class": "RelPos1dMultiHeadAttention", "params": {"num_heads": num_heads, "attn_drop_rate": attn_drop_rate, "num_pos_embeddings": max_pos_encoding, "weight_init": "default", "bias_init": "default"}}
            ],
            conv_params={ "class": "Conv1d", "params": {"padding": "same", "kernel_size": kernel_size}},
            ff_ratio=4,
            drop_rate=drop_rate,
            pos_embedding=None, 
            mask=attentions.Mask(), 
            conv_stride=2, 
            batch_norm=True,
            loss_prefix=loss_prefix
        )

        # Head
        self.head = layers.Linear(dim_model[-1], vocab_size) if include_head else nn.Identity()

    def forward(self, x, lengths):

        # Audio Preprocessing 
        x, lengths = self.audio_preprocessing(x, lengths)

        # Spec Augment
        x = self.spec_augment(x, lengths)

        # Unsqueeze 
        x = self.unsqueeze(x)

        # Stem
        x, lengths = self.subsampling_module(x, lengths)

        # Reshape
        x = self.reshape(x)

        # Transpose
        x = self.transpose(x)

        # Linear Proj
        x = self.linear(x)

        # Conformer
        x, lengths, interctc_outputs = self.back_end(x, lengths)

        # Head
        x = self.head(x)

        return x, lengths, interctc_outputs

class VisualEfficientConformerEncoder(nn.Module):

    def __init__(self, include_head=True, vocab_size=256, interctc_blocks=[3, 6, 9], num_blocks=[6, 6], loss_prefix="ctc"):
        super(VisualEfficientConformerEncoder, self).__init__()

        # Params
        dim_model=[256, 360]
        num_heads=4
        kernel_size=15
        drop_rate=0.1
        attn_drop_rate=0.0
        max_pos_encoding=10000

        # Stem 88 -> 44
        # MaxPool 44 -> 22 
        # ResNet 22 -> 11 -> 6 -> 3
        # GlobalPool 3 -> 1
        self.front_end = nn.Sequential(
            modules.ConvNeuralNetwork(
                dim_input=1,
                dim_layers=64,
                kernel_size=(5, 7, 7),
                strides=(1, 2, 2),
                norm="BatchNorm3d",
                act_fun="ReLU",
                drop_rate=0.0,
                dim=3
            ),
            layers.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding="same"),
            transforms.VideoToImages(), # (B, C, T, H, W) -> (BT, C, H, W)
            ResNet(include_stem=False, dim_output=dim_model[0], model="ResNet18")
        )

        self.expand_time = transforms.ImagesToVideos()

        # Conformer
        self.back_end = ConformerInterCTC(
            dim_model=dim_model,
            num_blocks=num_blocks,
            interctc_blocks=interctc_blocks,
            vocab_size=vocab_size,
            att_params={"class": "RelPos1dMultiHeadAttention", "params": {"num_heads": num_heads, "attn_drop_rate": attn_drop_rate, "num_pos_embeddings": max_pos_encoding, "weight_init": "default", "bias_init": "default"}},
            conv_params={"class": "Conv1d", "params": {"padding": "same", "kernel_size": kernel_size}},
            ff_ratio=4,
            drop_rate=drop_rate,
            pos_embedding=None, 
            mask=attentions.Mask(), 
            conv_stride=2, 
            batch_norm=True,
            loss_prefix=loss_prefix
        )

        # Head
        self.head = layers.Linear(dim_model[-1], vocab_size) if include_head else nn.Identity()

    def forward(self, x, lengths):

        # Frontend
        time = x.shape[2]
        x = self.front_end(x) # (B, C, T, H, W) -> (BT, C)
        x = x.unsqueeze(dim=-1).unsqueeze(dim=-1) # (BT, C) -> (BT, C, 1, 1)
        x = self.expand_time(x, time) # (BT, C, 1, 1) -> (B, C, T, 1, 1)
        x = x.squeeze(dim=-1).squeeze(dim=-1).transpose(1, 2) # (B, C, T, 1, 1) -> (B, T, C)

        # Backend
        x, lengths, interctc_outputs = self.back_end(x, lengths)

        # Head
        x = self.head(x)

        return x, lengths, interctc_outputs
    
class AudioVisualEfficientConformerEncoder(nn.Module):

    def __init__(self, include_head=True, vocab_size=256, v_interctc_blocks=[3, 6], a_interctc_blocks=[8, 11], f_interctc_blocks=[2]):
        super(AudioVisualEfficientConformerEncoder, self).__init__()

        # Params
        dim_model = 360
        num_blocks = 5
        num_heads = 4
        drop_rate = 0.1
        attn_drop_rate = 0.0
        max_pos_encoding = 10000
        kernel_size = 15
        v_num_blocks = [6, 1]
        a_num_blocks = [5, 6, 1]

        # Visual Encoder
        self.video_encoder = VisualEfficientConformerEncoder(include_head=False, vocab_size=vocab_size, interctc_blocks=v_interctc_blocks, num_blocks=v_num_blocks, loss_prefix="v_ctc")

        # Audio Encoder
        self.audio_encoder = AudioEfficientConformerEncoder(include_head=False, vocab_size=vocab_size, interctc_blocks=a_interctc_blocks, num_blocks=a_num_blocks, loss_prefix="a_ctc")

        # Fusion Module
        self.fusion_module = modules.FusionModule(a_dim_model=dim_model, v_dim_model=dim_model, f_dim_model=dim_model)

        # Audio-visual Encoder
        self.audio_visual_encoder = ConformerInterCTC(
            dim_model=dim_model,
            num_blocks=num_blocks,
            interctc_blocks=f_interctc_blocks,
            vocab_size=vocab_size,
            att_params={"class": "RelPos1dMultiHeadAttention", "params": {"num_heads": num_heads, "attn_drop_rate": attn_drop_rate, "num_pos_embeddings": max_pos_encoding, "weight_init": "default", "bias_init": "default"}},
            conv_params={"class": "Conv1d", "params": {"padding": "same", "kernel_size": kernel_size}},
            ff_ratio=4,
            drop_rate=drop_rate,
            pos_embedding=None, 
            mask=attentions.Mask(), 
            conv_stride=2, 
            batch_norm=True,
            loss_prefix="f_ctc"
        )

        # Head
        self.head = layers.Linear(dim_model, vocab_size) if include_head else nn.Identity()

    def forward(self, video, video_len, audio, audio_len):

        # Visual Encoder
        video, video_len, video_interctc_outputs = self.video_encoder(video, video_len)

        # Audio Encoder
        audio, audio_len, audio_interctc_outputs = self.audio_encoder(audio, audio_len)

        # Fusion Module
        x = self.fusion_module(audio, video)
        lengths = audio_len

        # Audio-visual Encoder
        x, lengths, interctc_outputs = self.audio_visual_encoder(x, lengths)
        interctc_outputs.update(video_interctc_outputs)
        interctc_outputs.update(audio_interctc_outputs)

        # Head
        x = self.head(x)

        return x, lengths, interctc_outputs

###############################################################################
# Adapter
###############################################################################

class AdapterMLP(nn.Module):

    """ Small MLP adapter that projects a frozen/pretrained encoder's feature space
    into the dimension expected by the downstream Conformer back-end.

    LayerNorm at the input is important here: pretrained encoders (wav2vec2, SwinLip)
    live in a different feature scale than the from-scratch Conformer stem did, and the
    rest of the network (InterCTC, fusion) was tuned assuming those scales.
    """

    def __init__(self, dim_input, dim_output, dim_hidden=None, drop_rate=0.1, act_fun="GELU"):
        super(AdapterMLP, self).__init__()

        dim_hidden = dim_hidden if dim_hidden is not None else max(dim_input, dim_output)

        self.norm = nn.LayerNorm(dim_input)
        self.layers = nn.Sequential(
            layers.Linear(dim_input, dim_hidden, weight_init="he_normal", bias_init="zeros"),
            getattr(nn, act_fun)(),
            nn.Dropout(p=drop_rate),
            layers.Linear(dim_hidden, dim_output, weight_init="he_normal", bias_init="zeros"),
        )

    def forward(self, x):

        # (B, T, Din) -> (B, T, Dout)
        x = self.norm(x)
        x = self.layers(x)

        return x

###############################################################################
# Audio: wav2vec 2.0
###############################################################################

class AudioWav2Vec2ConformerEncoder(nn.Module):

    """ Audio encoder built on a pretrained wav2vec 2.0 backbone.

    Replaces: AudioPreprocessing (mel spectrogram) + SpecAugment + ConvNeuralNetwork
    subsampling stem from the original AudioEfficientConformerEncoder.

    wav2vec2 already performs ~20ms / 50Hz feature extraction and self-supervised
    contextualization internally, so `num_blocks` / `interctc_blocks` here should
    generally be much shallower than the from-scratch audio branch — this is meant
    to be a thin Conformer head on top of already-strong features, not a full
    re-encoding stack.

    Input is a raw 16kHz waveform (B, T_samples), NOT a mel spectrogram.

    NOTE on `conv_stride`: ConformerInterCTC only applies it at a stage boundary
    (the last block of a stage, provided a following stage exists). A single-stage
    `num_blocks`/`dim_model` (e.g. [4] / [360], the default here) never downsamples
    regardless of `conv_stride`'s value. To actually downsample, use e.g.
    num_blocks=[2, 2], dim_model=[360, 360] with conv_stride=2.

    freeze_wav2vec2=True uses plain requires_grad_(False) + a train() override to
    keep it eval-locked (rather than register_module_buffer): register_module_buffer
    only correctly participates in device transfer when called directly on the
    Model instance you call .to()/.cuda() on -- nn.Module.to() moves parameters via
    an internal _apply() that recurses into children's _apply(), not their public
    .to(), so a buffer registered on a nested class like this one would silently
    never get moved to GPU. Normal registration + train() override sidesteps that:
    .to()/.cuda() work via standard recursion, and nn.Module.train() (unlike .to())
    does recurse via each child's public .train() method, so the override below is
    reliably invoked.

    unfreeze_last_n_layers>0 (only meaningful when freeze_wav2vec2=True) re-enables
    requires_grad on the LAST N transformer layers of wav2vec2.encoder.layers (HF's
    Wav2Vec2Model stack, indexed 0..num_hidden_layers-1) after the blanket freeze
    above, so the rest of the backbone (feature_extractor conv stack,
    feature_projection, and all-but-the-last-N encoder layers) stays frozen. The
    unfrozen layers are tracked in self.finetune_modules so the optimizer can be
    given a separate, much lower discriminative LR for them (see
    nnet/optimizers.py's get_discriminative_param_groups and
    AudioVisualPretrainedConformerInterCTC.compile() in models_zoo.py) -- do not
    just fold them into the main param group at the main schedule's LR, that risks
    distorting pretrained weights with updates sized for a from-scratch stack.
    """

    def __init__(
        self,
        include_head=True,
        vocab_size=256,
        interctc_blocks=[2],
        num_blocks=[4],
        dim_model=[360],
        loss_prefix="a_ctc",
        wav2vec2_name_or_path="facebook/wav2vec2-base-960h",
        freeze_wav2vec2=True,
        unfreeze_last_n_layers=0,
        adapter_dim_hidden=512,
        adapter_drop_rate=0.1,
        num_heads=4,
        kernel_size=15,
        drop_rate=0.1,
        attn_drop_rate=0.0,
        max_pos_encoding=10000,
        conv_stride=1,
    ):
        super(AudioWav2Vec2ConformerEncoder, self).__init__()

        if Wav2Vec2Model is None:
            raise ImportError(
                "transformers is required for AudioWav2Vec2ConformerEncoder: pip install transformers"
            )

        # Pretrained wav2vec2 backbone, (B, T_samples) -> (B, T', H)
        self.wav2vec2 = Wav2Vec2Model.from_pretrained(wav2vec2_name_or_path)

        self.freeze_wav2vec2 = freeze_wav2vec2
        self.finetune_modules = []
        if freeze_wav2vec2:
            self.wav2vec2.requires_grad_(False)
            self.wav2vec2.eval()

            if unfreeze_last_n_layers > 0:
                unfrozen_layers = self.wav2vec2.encoder.layers[-unfreeze_last_n_layers:]
                for layer in unfrozen_layers:
                    layer.requires_grad_(True)
                self.finetune_modules = list(unfrozen_layers)

        # Adapter, (B, T', H) -> (B, T', dim_model[0])
        self.adapter = AdapterMLP(
            dim_input=self.wav2vec2.config.hidden_size,
            dim_output=dim_model[0],
            dim_hidden=adapter_dim_hidden,
            drop_rate=adapter_drop_rate
        )

        # Shallow Conformer back-end + InterCTC on top of adapted wav2vec2 features
        self.back_end = ConformerInterCTC(
            dim_model=dim_model,
            num_blocks=num_blocks,
            interctc_blocks=interctc_blocks,
            vocab_size=vocab_size,
            att_params={"class": "RelPos1dMultiHeadAttention", "params": {"num_heads": num_heads, "attn_drop_rate": attn_drop_rate, "num_pos_embeddings": max_pos_encoding, "weight_init": "default", "bias_init": "default"}},
            conv_params={"class": "Conv1d", "params": {"padding": "same", "kernel_size": kernel_size}},
            ff_ratio=4,
            drop_rate=drop_rate,
            pos_embedding=None,
            mask=attentions.Mask(),
            conv_stride=conv_stride,   # keep at 1 unless you want to downsample below wav2vec2's native 50Hz
            batch_norm=True,
            loss_prefix=loss_prefix
        )

        # Head
        self.head = layers.Linear(dim_model[-1], vocab_size) if include_head else nn.Identity()

    @staticmethod
    def _lengths_to_attention_mask(x, lengths):
        # (B, Tmax) -> HF-style attention mask, 1 = valid sample, 0 = padding
        mask = torch.arange(x.shape[1], device=x.device)[None, :] < lengths[:, None]
        return mask.long()

    def train(self, mode=True):
        """ Keep wav2vec2 in eval mode even when the rest of the model is switched
        to train mode via model.train(). Without this, requires_grad=False alone
        does NOT stop wav2vec2's internal dropout / time-masking from behaving like
        training mode, and nn.Module.train() recursively re-enables train mode on
        every submodule every time it's called -- silently undoing the freeze each
        epoch otherwise.

        If unfreeze_last_n_layers>0, the blanket eval() below is applied first and
        then selectively overridden back to train() on self.finetune_modules --
        those layers have requires_grad=True and need real dropout/layer behavior
        during training, unlike the rest of the still-frozen backbone. """
        super(AudioWav2Vec2ConformerEncoder, self).train(mode)

        if self.freeze_wav2vec2:
            self.wav2vec2.eval()
            for layer in self.finetune_modules:
                layer.train(mode)

        return self

    def forward(self, x, lengths):

        # x: (B, T_samples) raw 16kHz waveform, NOT preprocessed to mel-spectrogram.
        # NOTE: wav2vec2 expects zero-mean / unit-variance normalized input per
        # utterance (see Wav2Vec2FeatureExtractor(do_normalize=True)). Apply that
        # normalization in your dataloader/collate_fn if you haven't already —
        # it is not reproduced here since it belongs with the rest of your audio
        # loading pipeline.
        attention_mask = self._lengths_to_attention_mask(x, lengths)

        # wav2vec2, (B, T_samples) -> (B, T', H)
        x = self.wav2vec2(x, attention_mask=attention_mask).last_hidden_state

        # Updated seq lengths after wav2vec2's internal conv downsampling
        lengths = self.wav2vec2._get_feat_extract_output_lengths(lengths).to(lengths.device)

        # Adapter, (B, T', H) -> (B, T', D)
        x = self.adapter(x)

        # Conformer back-end
        x, lengths, interctc_outputs = self.back_end(x, lengths)

        # Head
        x = self.head(x)

        return x, lengths, interctc_outputs

###############################################################################
# Visual: SwinLip
###############################################################################

class SwinLipFeatureExtractor(nn.Module):

    """ Wraps the official pretrained SwinLip model (github.com/park-ing/SwinLip,
    Park et al., Neurocomputing 2025 / arXiv:2505.04394) and exposes its per-frame
    visual embeddings instead of its word-classification logits.

    SwinLip.forward() is: frontend3D -> threeD_to_2D_tensor -> trunk -> tcn, where
    `tcn` is a DenseTCN word classifier trained for LRW's word-level task. We only
    want what comes out of `trunk`: a (B, T, 512) sequence, one 512-d embedding per
    input frame. `tcn` is dropped entirely here (it is unused, verified from the
    repo's models/swinlip.py — both frontend3D (stride 1 in time) and
    trunk/forward_features (reshapes back to (B, T, C) without any temporal pooling)
    preserve the input frame count, so with your 25fps LRS2/LRS3 crops this yields
    exactly 25 embeddings/sec, matching what you described.

    Input must be grayscale lip crops (B, 1, T, 88, 88), normalized the same way as
    the original repo's evaluation pipeline (utils/preprocess_inference.py):
        x_norm = (x / 255.0 - 0.421) / 0.165
    That normalization is NOT applied inside this wrapper — do it in your
    dataloader/collate_fn so it's easy to keep in sync with the rest of your
    LRS2/LRS3 preprocessing. Worth double-checking that whatever normalization
    nnet.datasets.LRS currently applies for video matches this convention, since it
    was presumably tuned for your old ResNet front-end instead.

    Stays a plain nn.Module -- freezing happens one level up, in
    VisualSwinLipConformerEncoder, which is where this gets consumed.
    """

    def __init__(self, config_path, checkpoint_path, map_location="cpu"):
        super(SwinLipFeatureExtractor, self).__init__()

        if SwinLip is None:
            raise ImportError(
                "Could not import SwinLip. Clone https://github.com/park-ing/SwinLip "
                "and add it to your PYTHONPATH before importing this module."
            )

        with open(config_path, "r") as f:
            config = json.load(f)

        self.model = SwinLip(config)

        checkpoint = torch.load(checkpoint_path, map_location=map_location)
        state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
        self.model.load_state_dict(state_dict)

        # Drop the word-classification head (DenseTCN) — we never call it, and it's
        # a non-trivial number of parameters to be carrying around for nothing.
        del self.model.tcn

        self.dim_output = self.model.backend_out  # 512

    def forward(self, x):

        # x: (B, 1, T, 88, 88), pre-normalized grayscale lip crops
        B, C, T, H, W = x.shape

        # 3D conv frontend (stride 1 in time -> T unchanged)
        x = self.model.frontend3D(x)
        Tnew = x.shape[2]

        # (B, C, T, H, W) -> (B*T, C, H, W)
        x = threeD_to_2D_tensor(x)

        # Swin Transformer stages + SwinLip's internal 2-layer Conformer,
        # reshaped back to (B, T, 512)
        x = self.model.trunk(x, B, Tnew)

        return x

class VisualSwinLipConformerEncoder(nn.Module):

    """ Visual encoder built on a pretrained SwinLip backbone
    (Park et al., "SwinLip: An Efficient Visual Speech Encoder for Lip Reading
    Using Swin Transformer", arXiv:2505.04394 / Neurocomputing 2025).

    Replaces: Conv3D stem + MaxPool3d + ResNet18 front-end from the original
    VisualEfficientConformerEncoder.

    `swinlip_backbone` should be a `SwinLipFeatureExtractor` instance (or anything
    exposing the same `.dim_output` attribute and `(B,C,T,H,W) -> (B,T,dim_output)`
    forward). `temporal_stride` defaults to 1 since SwinLip preserves the input
    frame rate — for 25fps LRS2/LRS3 crops that means 25 embeddings/sec, same as
    the video's native rate, with no length correction needed.

    freeze_backbone=True uses plain requires_grad_(False) + a train() override
    rather than register_module_buffer -- see AudioWav2Vec2ConformerEncoder's
    docstring for why (register_module_buffer only correctly participates in
    device transfer when registered directly on the Model instance .to()/.cuda()
    is called on, not on a nested class like this one). This matters even more
    here than for wav2vec2: frontend3D contains BatchNorm3d layers that would
    otherwise keep updating running mean/var from your batches purely from being
    in train() mode, independent of requires_grad.

    unfreeze_modules (only meaningful when freeze_backbone=True) lets you re-enable
    requires_grad on specific submodules of the backbone after the blanket freeze,
    given as dotted attribute paths resolved from `swinlip_backbone` itself (e.g.
    "model.trunk.layers.3" -- whatever the actual last-stage path turns out to be
    in your SwinLip checkout). Unlike wav2vec2's HF Wav2Vec2Model, SwinLip
    (github.com/park-ing/SwinLip) has no standardized/documented submodule naming
    this codebase can rely on, so there's no equivalent `unfreeze_last_n_layers`
    shortcut here -- inspect the actual module tree first (e.g. via the existing
    Model.show_modules() helper, or `print(swinlip_backbone)`) to find the correct
    path(s) for whatever counts as "the last block" in that repo's SwinLip class,
    then pass them here. Each resolved module is added to self.finetune_modules,
    same convention as AudioWav2Vec2ConformerEncoder, for the optimizer to give a
    separate discriminative LR (see nnet/optimizers.py).
    """

    def __init__(
        self,
        swinlip_backbone,
        include_head=True,
        vocab_size=256,
        interctc_blocks=[2],
        num_blocks=[4],
        dim_model=[360],
        loss_prefix="v_ctc",
        freeze_backbone=True,
        unfreeze_modules=[],
        temporal_stride=1,
        adapter_dim_hidden=512,
        adapter_drop_rate=0.1,
        num_heads=4,
        kernel_size=15,
        drop_rate=0.1,
        attn_drop_rate=0.0,
        max_pos_encoding=10000,
        conv_stride=1,
    ):
        super(VisualSwinLipConformerEncoder, self).__init__()

        assert hasattr(swinlip_backbone, "dim_output"), \
            "swinlip_backbone must expose a `dim_output` attribute (per-timestep feature dim)"

        self.temporal_stride = temporal_stride
        self.freeze_backbone = freeze_backbone

        # Pretrained SwinLip backbone, (B, C, T, H, W) -> (B, T', dim_output)
        self.front_end = swinlip_backbone
        self.finetune_modules = []
        if freeze_backbone:
            self.front_end.requires_grad_(False)
            self.front_end.eval()

            for dotted_path in unfreeze_modules:
                module = self.front_end
                for part in dotted_path.split("."):
                    module = getattr(module, part)
                module.requires_grad_(True)
                self.finetune_modules.append(module)

        # Adapter, (B, T', dim_output) -> (B, T', dim_model[0])
        self.adapter = AdapterMLP(
            dim_input=self.front_end.dim_output,
            dim_output=dim_model[0],
            dim_hidden=adapter_dim_hidden,
            drop_rate=adapter_drop_rate
        )

        # Shallow Conformer back-end + InterCTC on top of adapted SwinLip features
        self.back_end = ConformerInterCTC(
            dim_model=dim_model,
            num_blocks=num_blocks,
            interctc_blocks=interctc_blocks,
            vocab_size=vocab_size,
            att_params={"class": "RelPos1dMultiHeadAttention", "params": {"num_heads": num_heads, "attn_drop_rate": attn_drop_rate, "num_pos_embeddings": max_pos_encoding, "weight_init": "default", "bias_init": "default"}},
            conv_params={"class": "Conv1d", "params": {"padding": "same", "kernel_size": kernel_size}},
            ff_ratio=4,
            drop_rate=drop_rate,
            pos_embedding=None,
            mask=attentions.Mask(),
            conv_stride=conv_stride,   # keep at 1 unless you want extra downsampling below SwinLip's native rate
            batch_norm=True,
            loss_prefix=loss_prefix
        )

        # Head
        self.head = layers.Linear(dim_model[-1], vocab_size) if include_head else nn.Identity()

    def train(self, mode=True):
        """ Keep the SwinLip backbone in eval mode even when the rest of the model
        is switched to train mode via model.train() -- see class docstring.
        Selectively re-enables train() on self.finetune_modules afterward, same
        reasoning as AudioWav2Vec2ConformerEncoder.train(). """
        super(VisualSwinLipConformerEncoder, self).train(mode)

        if self.freeze_backbone:
            self.front_end.eval()
            for module in self.finetune_modules:
                module.train(mode)

        return self

    def forward(self, x, lengths):

        # SwinLip, (B, C, T, H, W) -> (B, T', dim_output)
        x = self.front_end(x)

        # Updated seq lengths if SwinLip performs internal temporal downsampling
        if self.temporal_stride > 1:
            lengths = torch.div(lengths - 1, self.temporal_stride, rounding_mode="floor") + 1

        # Adapter, (B, T', dim_output) -> (B, T', D)
        x = self.adapter(x)

        # Conformer back-end
        x, lengths, interctc_outputs = self.back_end(x, lengths)

        # Head
        x = self.head(x)

        return x, lengths, interctc_outputs

###############################################################################
# Audio-Visual: SwinLip + wav2vec2
###############################################################################

class AudioVisualPretrainedConformerEncoder(nn.Module):

    """ Audio-visual efficient conformer with pretrained front-ends:
      - Visual: SwinLip + adapter MLP   (replaces Conv3D stem + ResNet18)
      - Audio:  wav2vec 2.0 + adapter MLP (replaces mel-spectrogram front-end)

    Fusion module and audio-visual back-end are unchanged from the original
    AudioVisualEfficientConformerEncoder.

    Frame-rate alignment: SwinLip preserves your video's native rate (25fps for
    LRS2/LRS3), while wav2vec2 always outputs at a fixed 50Hz (20ms/frame) — almost
    exactly 2x the video rate. `audio_conv_stride=2` below downsamples the audio
    branch's Conformer back-end by 2x so both branches land at ~25Hz before fusion.
    `_align_time` is kept as a safety net for the residual few-frame mismatches
    that come from padding/rounding at each branch's edges, not as the primary
    alignment mechanism.

    Expects video already in (B, C, T, H, W) format and audio as raw (B, T_samples)
    waveform -- matches the original AudioVisualEfficientConformerEncoder's
    contract. The (B, T, H, W, C) -> (B, C, T, H, W) permute your dataset needs
    happens one level up, in the Model wrapper's forward() (see
    AudioVisualEfficientConformerInterCTC in models_zoo.py for the existing
    pattern -- same permute applies here).
    """

    def __init__(
        self,
        swinlip_backbone,
        include_head=True,
        vocab_size=256,
        v_interctc_blocks=[2],
        a_interctc_blocks=[2],
        f_interctc_blocks=[2],
        wav2vec2_name_or_path="facebook/wav2vec2-base-960h",
        freeze_swinlip=True,
        swinlip_temporal_stride=1,
        swinlip_unfreeze_modules=[],
        freeze_wav2vec2=True,
        wav2vec2_unfreeze_last_n_layers=0,
        audio_conv_stride=2,
    ):
        super(AudioVisualPretrainedConformerEncoder, self).__init__()

        # Params
        dim_model = 360
        num_blocks = 5
        num_heads = 4
        drop_rate = 0.1
        attn_drop_rate = 0.0
        max_pos_encoding = 10000
        kernel_size = 15

        # Visual Encoder: SwinLip + adapter + shallow Conformer
        # temporal_stride=1: SwinLip does not downsample time (confirmed from
        # models/swinlip.py — frontend3D uses stride=1 in T, and forward_features
        # reshapes back to (B, T, C) without any temporal pooling).
        self.video_encoder = VisualSwinLipConformerEncoder(
            swinlip_backbone=swinlip_backbone,
            include_head=False,
            vocab_size=vocab_size,
            interctc_blocks=v_interctc_blocks,
            num_blocks=[4],
            dim_model=[dim_model],
            loss_prefix="v_ctc",
            freeze_backbone=freeze_swinlip,
            unfreeze_modules=swinlip_unfreeze_modules,
            temporal_stride=swinlip_temporal_stride,
        )

        # Audio Encoder: wav2vec2 + adapter + shallow Conformer
        # conv_stride=2 brings wav2vec2's fixed 50Hz output down to ~25Hz to match
        # SwinLip's (native video-rate) output before fusion. NOTE: ConformerInterCTC
        # only applies conv_stride at a stage boundary (last block of a stage, when a
        # following stage exists) — a single-element num_blocks list never triggers a
        # downsample no matter what conv_stride is set to. Two same-width stages
        # ([2, 2]) give exactly one downsampling point between them.
        self.audio_encoder = AudioWav2Vec2ConformerEncoder(
            include_head=False,
            vocab_size=vocab_size,
            interctc_blocks=a_interctc_blocks,
            num_blocks=[2, 2] if audio_conv_stride > 1 else [4],
            dim_model=[dim_model, dim_model] if audio_conv_stride > 1 else [dim_model],
            loss_prefix="a_ctc",
            wav2vec2_name_or_path=wav2vec2_name_or_path,
            freeze_wav2vec2=freeze_wav2vec2,
            unfreeze_last_n_layers=wav2vec2_unfreeze_last_n_layers,
            conv_stride=audio_conv_stride,
        )

        # Fusion Module (unchanged)
        self.fusion_module = modules.FusionModule(a_dim_model=dim_model, v_dim_model=dim_model, f_dim_model=dim_model)

        # Combined list of any partially-unfrozen pretrained-backbone submodules
        # across both branches, for the optimizer to give a separate discriminative
        # LR group (see AudioVisualPretrainedConformerInterCTC.compile() in
        # models_zoo.py and nnet/optimizers.get_discriminative_param_groups). Empty
        # when both backbones are fully frozen (the original behavior).
        self.finetune_modules = self.video_encoder.finetune_modules + self.audio_encoder.finetune_modules

        # Audio-visual Encoder (unchanged)
        self.audio_visual_encoder = ConformerInterCTC(
            dim_model=dim_model,
            num_blocks=num_blocks,
            interctc_blocks=f_interctc_blocks,
            vocab_size=vocab_size,
            att_params={"class": "RelPos1dMultiHeadAttention", "params": {"num_heads": num_heads, "attn_drop_rate": attn_drop_rate, "num_pos_embeddings": max_pos_encoding, "weight_init": "default", "bias_init": "default"}},
            conv_params={"class": "Conv1d", "params": {"padding": "same", "kernel_size": kernel_size}},
            ff_ratio=4,
            drop_rate=drop_rate,
            pos_embedding=None,
            mask=attentions.Mask(),
            conv_stride=2,
            batch_norm=True,
            loss_prefix="f_ctc"
        )

        # Head
        self.head = layers.Linear(dim_model, vocab_size) if include_head else nn.Identity()

    @staticmethod
    def _align_time(a, a_lengths, b, b_lengths):
        """ Align two (B, T, D) sequences with potentially different frame rates
        to a common time axis by linearly interpolating the longer one down to
        match the shorter one. Returns (a, b, lengths). """

        Ta, Tb = a.shape[1], b.shape[1]

        if Ta == Tb:
            return a, b, torch.minimum(a_lengths, b_lengths)

        T = min(Ta, Tb)

        if Ta != T:
            a = F.interpolate(a.transpose(1, 2), size=T, mode="linear", align_corners=False).transpose(1, 2)
            a_lengths = torch.clamp((a_lengths.float() * T / Ta).round().long(), max=T)

        if Tb != T:
            b = F.interpolate(b.transpose(1, 2), size=T, mode="linear", align_corners=False).transpose(1, 2)
            b_lengths = torch.clamp((b_lengths.float() * T / Tb).round().long(), max=T)

        lengths = torch.minimum(a_lengths, b_lengths)

        return a, b, lengths

    def forward(self, video, video_len, audio, audio_len):

        # Visual Encoder
        video, video_len, video_interctc_outputs = self.video_encoder(video, video_len)

        # Audio Encoder
        audio, audio_len, audio_interctc_outputs = self.audio_encoder(audio, audio_len)

        # Align frame rates before fusion (see class docstring)
        audio, video, lengths = self._align_time(audio, audio_len, video, video_len)

        # Fusion Module
        x = self.fusion_module(audio, video)

        # Audio-visual Encoder
        x, lengths, interctc_outputs = self.audio_visual_encoder(x, lengths)
        interctc_outputs.update(video_interctc_outputs)
        interctc_outputs.update(audio_interctc_outputs)

        # Head
        x = self.head(x)

        return x, lengths, interctc_outputs

###############################################################################
# Example usage
###############################################################################
#
# import sys
# sys.path.append("/path/to/SwinLip")  # github.com/park-ing/SwinLip clone
#
# swinlip = SwinLipFeatureExtractor(
#     config_path="/path/to/SwinLip/configs/swinlip.json",
#     checkpoint_path="/path/to/SwinLip/checkpoints/lrw_swinlip.pth",
# )
#
# model = AudioVisualPretrainedConformerEncoder(
#     swinlip_backbone=swinlip,
#     freeze_swinlip=True,
#     freeze_wav2vec2=True,
# )
#
# # video: (B, 1, T_v, 88, 88) normalized grayscale lip crops @ 25fps
# # audio: (B, T_a) raw 16kHz waveform, zero-mean/unit-variance normalized
# logits, lengths, interctc_outputs = model(video, video_len, audio, audio_len)
#
# # No optimizer filtering needed: requires_grad=False on the frozen backbones
# # means torch.optim.* (and this codebase's Adam/AdamW/SGD wrappers in
# # nnet/optimizers.py, which just call through to the standard step() loop)
# # skip params with grad=None during .step() -- confirmed from optimizers.py,
# # nothing unusual there.