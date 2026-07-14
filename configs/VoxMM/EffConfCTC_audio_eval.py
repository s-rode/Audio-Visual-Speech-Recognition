import os
from pathlib import Path
import sys
sys.path.append("../../")

import nnet
import torch
from configs.VoxMM.ctc_decoding import decoder_options_from_env, make_ctc_decoder


def _env_or_existing_path(env_name, local_path, cluster_path):
    value = os.environ.get(env_name)
    if value:
        return value
    if Path(local_path).exists():
        return local_path
    return cluster_path


vocab_size = 256
interctc_blocks = []
loss_weights = None
att_type = "patch"

tokenizer_path = "datasets/LRS3/tokenizerbpe256.model"
pretrained_checkpoint = os.environ.get(
    "AVEC_AUDIO_CHECKPOINT",
    "callbacks/LRS23/AO/EffConfCTC/checkpoints_swa-equal-200-210.ckpt",
)
manifest_eval = _env_or_existing_path(
    "VOXMM_EVAL_MANIFEST",
    "../../data/voxmm_dev_eval/manifest.local.csv",
    "/hpcwork/ww202463/team-1-data/lrs_voxmm/manifest_dev_eval_avec_roi/manifest.csv",
)
callback_path = os.environ.get("VOXMM_CALLBACK_PATH", "callbacks/VoxMM/AO/EffConfCTC_eval")

batch_size = int(os.environ.get("VOXMM_EVAL_BATCH_SIZE", os.environ.get("VOXMM_BATCH_SIZE", "8")))
precision = torch.float32
recompute_metrics = True
beam_search = False  # Kept for compatibility with upstream config naming.
decoder_options = decoder_options_from_env()

model = nnet.AudioEfficientConformerInterCTC(
    vocab_size=vocab_size,
    att_type=att_type,
    interctc_blocks=interctc_blocks,
)
model.compile(
    losses=nnet.CTCLoss(zero_infinity=True, assert_shorter=False),
    metrics=nnet.WordErrorRate(),
    decoders=make_ctc_decoder(nnet, tokenizer_path, decoder_options),
    loss_weights=loss_weights,
)
checkpoint = torch.load(pretrained_checkpoint, map_location=model.device)
model.load_state_dict(checkpoint["model_state_dict"], strict=True)
print("Loaded pretrained audio checkpoint:", pretrained_checkpoint)


import types
import wandb


def _metric_value(value):
    return float(value.detach().cpu()) if hasattr(value, "detach") else float(value)


wandb.init(
    project=os.environ.get("WANDB_PROJECT", "team1-asr"),
    name=os.environ.get("WANDB_RUN_NAME", "avec-voxmm-audio-only-baseline"),
    config={
        "model": "AudioEfficientConformerInterCTC",
        "modality": "audio-only",
        "pretrained": pretrained_checkpoint,
        "eval_manifest": manifest_eval,
        "batch_size": batch_size,
        "decoder": decoder_options["decoder"],
        "beam_size": decoder_options["beam_size"],
        "beam_cutoff_top_n": decoder_options["beam_cutoff_top_n"],
        "beam_temperature": decoder_options["beam_temperature"],
        "beam_token_bonus": decoder_options["beam_token_bonus"],
        "lm_manifest": decoder_options["lm_manifest"],
        "lm_type": decoder_options["lm_type"],
        "lm_weight": decoder_options["lm_weight"],
        "lm_order": decoder_options["lm_order"],
        "lm_smoothing": decoder_options["lm_smoothing"],
        "lm_discount": decoder_options["lm_discount"],
        "word_bonus": decoder_options["word_bonus"],
        "kenlm_path": decoder_options["kenlm_path"],
        "unigram_path": decoder_options["unigram_path"],
        "kenlm_max_words": decoder_options["kenlm_max_words"],
        "dataset": os.environ.get("VOXMM_DATASET_NAME", "VoxMM-dev-eval"),
    },
)

_original_evaluate = model.evaluate.__func__


def _wandb_evaluate(self, dataset_eval, eval_steps=None, verbose=0, recompute_metrics=False):
    losses, metrics = _original_evaluate(self, dataset_eval, eval_steps, verbose, recompute_metrics)
    if self.rank == 0:
        log_dict = {f"Evaluation/0/{key}": _metric_value(value) for key, value in losses.items()}
        log_dict.update({f"Evaluation/0/{key}": _metric_value(value) for key, value in metrics.items()})
        log_dict.update({f"eval_{key}": _metric_value(value) for key, value in losses.items()})
        log_dict.update({f"eval_{key}": _metric_value(value) for key, value in metrics.items()})
        wandb.log(log_dict, step=int(getattr(self, "model_step", 0)))
        wandb.run.summary.update(log_dict)
    return losses, metrics


model.evaluate = types.MethodType(_wandb_evaluate, model)


collate_fn = nnet.CollateFn(
    inputs_params=[{"axis": 1, "padding": True}, {"axis": 4}],
    targets_params=({"axis": 2, "padding": True}, {"axis": 5}),
)
evaluation_dataset = nnet.datasets.VoxMM(
    batch_size=batch_size,
    collate_fn=collate_fn,
    manifest_path=manifest_eval,
    tokenizer_path=tokenizer_path,
    shuffle=False,
    load_audio=True,
    load_video=False,
    align=False,
)
