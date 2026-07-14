#!/usr/bin/env python3
"""Sweep Flashlight decoder settings over cached VoxMM CTC emissions."""

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import jiwer
import sentencepiece as spm
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
AVEC_ROOT = REPO_ROOT / "external" / "AVEC"
sys.path.insert(0, str(AVEC_ROOT))

from configs.VoxMM.ctc_decoding import (  # noqa: E402
    FlashlightCTCDecoder,
    KneserNeyWordNGramLM,
)


DEFAULT_SETTINGS = [
    "w050_bm05:0.50:-0.50:50:50:50:false:1.0",
    "w065_bm05:0.65:-0.50:50:50:50:false:1.0",
    "w080_bm05:0.80:-0.50:50:50:50:false:1.0",
    "w095_bm05:0.95:-0.50:50:50:50:false:1.0",
    "w080_bm025:0.80:-0.25:50:50:50:false:1.0",
    "w080_bm075:0.80:-0.75:50:50:50:false:1.0",
]


def parse_bool(value):
    return value.lower() in {"1", "true", "yes", "on"}


def parse_setting(value):
    fields = value.split(":")
    if len(fields) not in {8, 10, 11}:
        raise argparse.ArgumentTypeError(
            "setting must be name:lm_weight:word_bonus:beam:token_cutoff:"
            "beam_threshold:log_add:temperature[:rescore_lm_weight:"
            "rescore_word_bonus[:blank_bias]]"
        )
    setting = {
        "name": fields[0],
        "lm_weight": float(fields[1]),
        "word_bonus": float(fields[2]),
        "beam_size": int(fields[3]),
        "token_cutoff": int(fields[4]),
        "beam_threshold": float(fields[5]),
        "log_add": parse_bool(fields[6]),
        "temperature": float(fields[7]),
    }
    if len(fields) >= 10:
        setting["rescore_lm_weight"] = float(fields[8])
        setting["rescore_word_bonus"] = float(fields[9])
    if len(fields) == 11:
        setting["blank_bias"] = float(fields[10])
    return setting


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "cache_dirs",
        nargs="+",
        help="One cache directory, or several checkpoint caches to average",
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--lm-manifest", required=True)
    parser.add_argument("--kenlm")
    parser.add_argument(
        "--tokenizer",
        default=str(AVEC_ROOT / "datasets/LRS3/tokenizerbpe256.model"),
    )
    parser.add_argument("--unigrams")
    parser.add_argument("--lexicon-unigrams")
    parser.add_argument("--exclude-word", action="append", default=[])
    parser.add_argument("--lexicon-free", action="store_true")
    parser.add_argument("--max-words", type=int, default=200000)
    parser.add_argument("--lm-order", type=int, default=4)
    parser.add_argument("--lm-discount", type=float, default=0.75)
    parser.add_argument("--blank-bias", type=float, default=0.0)
    parser.add_argument("--nbest", type=int, default=1)
    parser.add_argument("--rescore-lm-weight", type=float, default=0.0)
    parser.add_argument("--rescore-word-bonus", type=float, default=0.0)
    parser.add_argument("--save-nbest", action="store_true")
    parser.add_argument(
        "--cache-weights",
        help="Comma-separated interpolation weights for multiple cache directories",
    )
    parser.add_argument(
        "--setting",
        action="append",
        type=parse_setting,
        help=(
            "name:lm_weight:word_bonus:beam:token_cutoff:"
            "beam_threshold:log_add:temperature[:rescore_lm_weight:"
            "rescore_word_bonus[:blank_bias]]"
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=str(REPO_ROOT / "outputs/voxmm_cached_decoder_sweep"),
    )
    args = parser.parse_args()
    if not args.lexicon_free and not args.kenlm:
        parser.error("--kenlm is required unless --lexicon-free is set")
    return args


def load_references(manifest_path, tokenizer_path):
    tokenizer = spm.SentencePieceProcessor(tokenizer_path)
    rows = []
    with Path(manifest_path).open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            rows.append(
                {
                    "utterance_id": row["utterance_id"],
                    "reference": tokenizer.decode(
                        tokenizer.encode(row["reference"].lower())
                    ),
                }
            )
    return rows


def cache_files(cache_dirs):
    files_by_cache = [sorted(Path(path).glob("batch_*.pt")) for path in cache_dirs]
    if not files_by_cache or not files_by_cache[0]:
        raise FileNotFoundError(f"No batch_*.pt files found in {cache_dirs[0]}")
    expected = len(files_by_cache[0])
    if any(len(files) != expected for files in files_by_cache):
        counts = [len(files) for files in files_by_cache]
        raise ValueError(f"Emission caches have different batch counts: {counts}")
    return list(zip(*files_by_cache))


def parse_cache_weights(value, count):
    weights = [1.0] * count if value is None else [float(item) for item in value.split(",")]
    if len(weights) != count:
        raise ValueError(f"Expected {count} cache weights, got {len(weights)}")
    total = sum(weights)
    if total == 0:
        raise ValueError("Cache weights must not sum to zero")
    return [weight / total for weight in weights]


def load_ensemble(paths, weights):
    payloads = [torch.load(path, map_location="cpu", weights_only=True) for path in paths]
    reference_shape = payloads[0]["logits"].shape
    reference_lengths = payloads[0]["lengths"]
    for path, payload in zip(paths[1:], payloads[1:]):
        if payload["logits"].shape != reference_shape:
            raise ValueError(f"Logit shape mismatch in {path}")
        if not torch.equal(payload["lengths"], reference_lengths):
            raise ValueError(f"Length mismatch in {path}")
    logits = sum(
        payload["logits"].float() * weight
        for payload, weight in zip(payloads, weights)
    )
    return logits, reference_lengths


def error_counts(references, predictions):
    result = jiwer.process_words(references, predictions)
    return {
        "hits": result.hits,
        "substitutions": result.substitutions,
        "deletions": result.deletions,
        "insertions": result.insertions,
    }


def word_error_count(reference, prediction):
    result = jiwer.process_words(reference, prediction)
    return result.substitutions + result.deletions + result.insertions


def main():
    args = parse_args()
    settings = args.setting or [parse_setting(value) for value in DEFAULT_SETTINGS]
    if args.save_nbest and len(settings) != 1:
        raise ValueError("--save-nbest requires exactly one --setting")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = load_references(args.manifest, args.tokenizer)
    grouped_cache_files = cache_files(args.cache_dirs)
    weights = parse_cache_weights(args.cache_weights, len(args.cache_dirs))

    lm = KneserNeyWordNGramLM(
        manifest_path=args.lm_manifest,
        order=args.lm_order,
        discount=args.lm_discount,
    )
    results = []
    for setting in settings:
        started = time.perf_counter()
        rescore_lm_weight = setting.get(
            "rescore_lm_weight", args.rescore_lm_weight
        )
        rescore_word_bonus = setting.get(
            "rescore_word_bonus", args.rescore_word_bonus
        )
        blank_bias = setting.get("blank_bias", args.blank_bias)
        decoder = FlashlightCTCDecoder(
            tokenizer_path=args.tokenizer,
            lm=lm,
            beam_size=setting["beam_size"],
            cutoff_top_n=setting["token_cutoff"],
            temperature=setting["temperature"],
            blank_bias=blank_bias,
            lm_weight=setting["lm_weight"],
            word_bonus=setting["word_bonus"],
            asset_dir=str(output_dir / f"assets_{setting['name']}"),
            kenlm_path=args.kenlm,
            unigram_path=args.unigrams,
            lexicon_unigram_path=args.lexicon_unigrams,
            excluded_words=args.exclude_word,
            lexicon_free=args.lexicon_free,
            max_words=args.max_words,
            beam_threshold=setting["beam_threshold"],
            log_add=setting["log_add"],
            nbest=args.nbest,
            rescore_lm_weight=rescore_lm_weight,
            rescore_word_bonus=rescore_word_bonus,
        )
        predictions = []
        candidate_lists = [] if args.save_nbest else None
        with torch.inference_mode():
            for paths in grouped_cache_files:
                logits, lengths = load_ensemble(paths, weights)
                if candidate_lists is None:
                    predictions.extend(decoder((logits, lengths)))
                else:
                    batch_candidates = decoder.decode_candidates(logits, lengths)
                    candidate_lists.extend(batch_candidates)
                    predictions.extend(
                        candidates[0]["text"] if candidates else ""
                        for candidates in batch_candidates
                    )

        if len(predictions) != len(rows):
            raise ValueError(
                f"Decoded {len(predictions)} samples but manifest contains {len(rows)}"
            )
        references = [row["reference"] for row in rows]
        wer = 100.0 * jiwer.wer(references, predictions)
        result = {
            **setting,
            "rescore_lm_weight": rescore_lm_weight,
            "rescore_word_bonus": rescore_word_bonus,
            "blank_bias": blank_bias,
            "wer": wer,
            **error_counts(references, predictions),
            "seconds": time.perf_counter() - started,
        }
        if candidate_lists is not None:
            oracle_predictions = [
                min(
                    candidates,
                    key=lambda candidate: word_error_count(
                        reference, candidate["text"]
                    ),
                )["text"]
                if candidates
                else ""
                for reference, candidates in zip(references, candidate_lists)
            ]
            result["oracle_wer"] = 100.0 * jiwer.wer(
                references, oracle_predictions
            )
        results.append(result)
        print(
            f"{setting['name']}: WER={wer:.4f}% "
            f"(S={result['substitutions']}, D={result['deletions']}, "
            f"I={result['insertions']}, {result['seconds']:.1f}s)"
        )

        prediction_path = output_dir / f"{setting['name']}.jsonl"
        with prediction_path.open("w", encoding="utf-8") as handle:
            for row, prediction in zip(rows, predictions):
                handle.write(
                    json.dumps({**row, "prediction": prediction}, ensure_ascii=False)
                    + "\n"
                )

        if candidate_lists is not None:
            candidate_path = output_dir / f"{setting['name']}_nbest.jsonl"
            with candidate_path.open("w", encoding="utf-8") as handle:
                for row, candidates in zip(rows, candidate_lists):
                    handle.write(
                        json.dumps(
                            {**row, "candidates": candidates},
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
            print(
                f"{setting['name']}: oracle WER="
                f"{result['oracle_wer']:.4f}%"
            )
            print(f"Wrote {candidate_path}")

    summary = {
        "cache_dirs": args.cache_dirs,
        "cache_weights": weights,
        "manifest": args.manifest,
        "lm_manifest": args.lm_manifest,
        "kenlm": args.kenlm,
        "lexicon_free": args.lexicon_free,
        "excluded_words": args.exclude_word,
        "nbest": args.nbest,
        "rescore_lm_weight": args.rescore_lm_weight,
        "rescore_word_bonus": args.rescore_word_bonus,
        "blank_bias": args.blank_bias,
        "num_utterances": len(rows),
        "results": sorted(results, key=lambda item: item["wer"]),
    }
    summary_path = output_dir / "results.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
