# Vendored code and checkpoints

This tree bundles two upstream repos as plain files (no git submodule, no
nested `.git`) rather than forking them separately, to keep everything in one
place in `team-1/AVSR`.

## Vendored code

- **`AVEC/`** -- vendored from [github.com/burchim/AVEC](https://github.com/burchim/AVEC),
  commit `1bf8ad079b90af383090f1584a20fe9467f98f47` (branch `master`), with
  substantial local modifications for AVEC++: `nnet/models_zoo.py` and
  `nnet/networks.py` carry the SwinLip/wav2vec2 pretrained-encoder
  integration and InterCTC changes, plus smaller edits to `nnet/blocks.py`,
  `nnet/layers.py`, `nnet/metrics.py`, `nnet/optimizers.py`,
  `nnet/transforms.py`, and `requirements.txt`. Everything else is
  unmodified upstream AVEC.
- **`SwinLip/SwinLip/`** -- vendored from
  [github.com/park-ing/SwinLip](https://github.com/park-ing/SwinLip),
  commit `0be70b224e42619f72ad283e85303f5e10544004` (branch `master`),
  unmodified.

If you need to diff against upstream later: clone the original repo at that
commit somewhere separate and diff directories, since git history for these
two isn't carried in `team-1/AVSR`.

## Checkpoints not tracked in git

Excluded via `.gitignore` (large binaries, ~1.4GB combined) -- fetch these
after cloning, before running anything:

**`wav2vec2/wav2vec2-base-960h/`** (~1.1GB) -- the localized HF snapshot used
by `wav2vec2_name_or_path` in the CLAIX/pretrained configs. Refetch with:

```bash
pip install -U "huggingface_hub[cli]"
huggingface-cli download facebook/wav2vec2-base-960h --local-dir wav2vec2/wav2vec2-base-960h
```

**`SwinLip/lrw_swinlip.pth`** (~245MB) -- the pretrained SwinLip backbone
checkpoint referenced by `swinlip_checkpoint_path` in those same configs.

Download link: https://drive.google.com/file/d/1t5IIUjEq3GYlgVmeGXyiHuqbBCCJ2ecX/view
