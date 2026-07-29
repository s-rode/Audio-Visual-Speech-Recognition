"""
Print the SwinLip backbone's actual submodule tree (dotted path + class name for
every nn.Module inside it), so you can find the correct path(s) to pass as
`swinlip_unfreeze_modules` in configs/EffConfInterCTC_pretrained_claix.py.

Why this is needed: unlike wav2vec2 (HF's Wav2Vec2Model has a documented, stable
`.encoder.layers` ModuleList -- see wav2vec2_unfreeze_last_n_layers), this repo's
SwinLip class (github.com/park-ing/SwinLip) has no submodule naming this codebase
assumes anything about. VisualSwinLipConformerEncoder.unfreeze_modules takes
dotted attribute paths resolved via plain getattr-chains (see its docstring in
nnet/networks.py) -- this script is how you find what those paths actually are
before setting them.

Only prints the architecture, doesn't run any data through it -- no dataset/GPU
needed, --cpu is fine and faster for this.

Usage (run from AVEC/ project root):
    python tools/inspect_swinlip_modules.py \
        --config_file configs/EffConfInterCTC_pretrained_claix.py \
        --load_last \
        --cpu

    # narrow the printout to modules whose dotted path contains a substring:
    python tools/inspect_swinlip_modules.py --config_file ... --load_last --cpu --filter trunk

Per SwinLipFeatureExtractor's docstring, SwinLip.forward() is:
    frontend3D -> threeD_to_2D_tensor -> trunk -> tcn
and trunk itself is "Swin Transformer stages + SwinLip's internal 2-layer
Conformer" -- so the modules you're looking for as the "last block(s)" are most
likely near the end of the trunk's listing below: either the last Swin stage, or
the trailing 2-layer Conformer (arguably the closer analogue to wav2vec2's
"last N transformer layers", since it's what feeds the adapter directly). Look at
both before picking.
"""

import argparse
import importlib
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import functions


def build_args(config_file, checkpoint, load_last, cpu):
    """ Same minimal Namespace stand-in as tools/inspect_training_samples.py. """
    return argparse.Namespace(
        rank=0,
        distributed=False,
        parallel=False,
        cpu=cpu,
        config_file=config_file,
        checkpoint=checkpoint,
        load_last=load_last,
        world_size=1,
        num_workers=0,
        show_dict=False,
        show_modules=False,
    )


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--load_last", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--filter", default=None, help="Only print dotted paths containing this substring (case-insensitive)")
    cli_args = parser.parse_args()

    if cli_args.checkpoint is None and not cli_args.load_last:
        parser.error("Pass --checkpoint <name> or --load_last")

    args = build_args(cli_args.config_file, cli_args.checkpoint, cli_args.load_last, cli_args.cpu)
    args.config = importlib.import_module(args.config_file.replace(".py", "").replace("/", "."))
    model = functions.load_model(args)
    model.eval()

    swinlip_front_end = model.encoder.video_encoder.front_end

    if cli_args.filter is None:
        print("Full SwinLip backbone module tree (model.encoder.video_encoder.front_end):\n")
        model.show_modules(module=swinlip_front_end)
    else:
        # show_modules() itself has no filter param -- reimplement its loop here
        # with the same "class: ..." formatting, just skipping non-matching keys.
        named_modules = dict(swinlip_front_end.named_modules())
        needle = cli_args.filter.lower()
        matches = {k: v for k, v in named_modules.items() if needle in k.lower()}
        print("SwinLip backbone modules matching '{}' ({} of {} total):\n".format(cli_args.filter, len(matches), len(named_modules)))
        for key, value in matches.items():
            print("{}  class: {}".format(key, type(value).__name__))

    print(
        "\nDotted paths above are relative to `front_end` -- pass them to "
        "swinlip_unfreeze_modules exactly as printed, e.g. "
        "swinlip_unfreeze_modules = [\"model.trunk.layers.3\"] "
        "(VisualSwinLipConformerEncoder resolves each one via a plain "
        "getattr-chain off `front_end`, see its docstring in nnet/networks.py)."
    )


if __name__ == "__main__":
    main()
