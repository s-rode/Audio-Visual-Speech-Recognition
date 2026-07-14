#!/usr/bin/env python3
"""Tune log-linear weights for pre-scored VoxMM N-best hypotheses."""

import argparse
import json
from pathlib import Path

try:
    from .voxmm_transformer_lm import (
        FILLED_PAUSE_WORDS,
        load_rows,
        normalize_hypothesis,
        word_errors,
    )
except ImportError:
    from voxmm_transformer_lm import (
        FILLED_PAUSE_WORDS,
        load_rows,
        normalize_hypothesis,
        word_errors,
    )


WEIGHT_NAMES = (
    "transformer_weight",
    "word_bonus",
    "forward_lm_weight",
    "reverse_lm_weight",
    "drop_penalty",
)


def parse_float_range(value):
    try:
        fields = value.split(":")
        if len(fields) != 2:
            raise ValueError
        lower, upper = (float(field) for field in fields)
    except (AttributeError, ValueError) as error:
        raise argparse.ArgumentTypeError(
            "range must be LOWER:UPPER"
        ) from error
    if lower > upper:
        raise argparse.ArgumentTypeError("range lower bound exceeds upper bound")
    return lower, upper


def parse_weights(value):
    try:
        fields = [float(field) for field in value.split(":")]
    except (AttributeError, ValueError) as error:
        raise argparse.ArgumentTypeError(
            "weights must contain five colon-separated numbers"
        ) from error
    if len(fields) != len(WEIGHT_NAMES):
        raise argparse.ArgumentTypeError(
            "weights must contain five colon-separated numbers"
        )
    return fields


def build_search_arrays(rows, top_k, dropped_words, np):
    candidate_rows = [row["candidates"][:top_k] for row in rows]
    if not candidate_rows or any(not candidates for candidates in candidate_rows):
        raise ValueError("Every utterance must contain at least one candidate")

    max_candidates = max(len(candidates) for candidates in candidate_rows)
    utterance_count = len(rows)
    base_scores = np.full(
        (utterance_count, max_candidates),
        -np.inf,
        dtype=np.float64,
    )
    features = np.zeros(
        (utterance_count, max_candidates, len(WEIGHT_NAMES)),
        dtype=np.float64,
    )
    errors = np.full(
        (utterance_count, max_candidates),
        np.iinfo(np.int32).max,
        dtype=np.int64,
    )
    normalized_texts = []
    reference_words = 0

    for row_index, (row, candidates) in enumerate(zip(rows, candidate_rows)):
        row_texts = []
        for candidate_index, candidate in enumerate(candidates):
            normalized, dropped_count = normalize_hypothesis(
                candidate["text"],
                dropped_words,
            )
            base_scores[row_index, candidate_index] = candidate["score"]
            features[row_index, candidate_index] = [
                candidate["transformer_lm_score"],
                len(normalized.split()),
                candidate.get("neural_lm_score", 0.0),
                candidate.get("aux_neural_lm_score", 0.0),
                -dropped_count,
            ]
            errors[row_index, candidate_index] = word_errors(
                row["reference"],
                normalized,
            )[0]
            row_texts.append(normalized)
        normalized_texts.append(row_texts)
        reference_words += len(row["reference"].split())

    return base_scores, features, errors, normalized_texts, reference_words


def search(args):
    import numpy as np

    rows = load_rows(args.candidates)
    dropped_words = {word.lower() for word in args.drop_word}
    if args.drop_filled_pauses:
        dropped_words.update(FILLED_PAUSE_WORDS)
    arrays = build_search_arrays(rows, args.top_k, dropped_words, np)
    base_scores, features, errors, normalized_texts, reference_words = arrays
    bounds = np.asarray(
        [
            args.transformer_range,
            args.word_bonus_range,
            args.forward_lm_range,
            args.reverse_lm_range,
            args.drop_penalty_range,
        ],
        dtype=np.float64,
    )
    global_lower = bounds[:, 0]
    global_upper = bounds[:, 1]
    rng = np.random.default_rng(args.seed)
    best_errors = np.iinfo(np.int64).max
    best_weights = None
    round_results = []

    for round_index in range(args.rounds):
        if round_index == 0 or best_weights is None:
            lower = global_lower
            upper = global_upper
        else:
            half_width = (
                (global_upper - global_lower)
                * args.refinement_fraction**round_index
            )
            lower = np.maximum(global_lower, best_weights - half_width)
            upper = np.minimum(global_upper, best_weights + half_width)

        sampled = rng.uniform(
            lower,
            upper,
            size=(args.samples, len(WEIGHT_NAMES)),
        )
        extras = list(args.anchor)
        if best_weights is not None:
            extras.append(best_weights)
        if extras:
            sampled = np.vstack([np.asarray(extras), sampled])

        round_best_errors = np.iinfo(np.int64).max
        round_best_weights = None
        for start in range(0, len(sampled), args.search_batch_size):
            weight_batch = sampled[start : start + args.search_batch_size]
            scores = base_scores[None, :, :] + np.einsum(
                "ukf,bf->buk",
                features,
                weight_batch,
            )
            selected = scores.argmax(axis=2)
            selected_errors = np.take_along_axis(
                np.broadcast_to(errors, scores.shape),
                selected[:, :, None],
                axis=2,
            ).squeeze(axis=2)
            totals = selected_errors.sum(axis=1)
            batch_best = int(totals.argmin())
            if int(totals[batch_best]) < round_best_errors:
                round_best_errors = int(totals[batch_best])
                round_best_weights = weight_batch[batch_best].copy()

        if round_best_errors < best_errors:
            best_errors = round_best_errors
            best_weights = round_best_weights
        result = {
            "round": round_index + 1,
            "errors": best_errors,
            "wer": 100.0 * best_errors / reference_words,
            "weights": dict(zip(WEIGHT_NAMES, best_weights.tolist())),
        }
        round_results.append(result)
        print(
            f"round={round_index + 1} WER={result['wer']:.4f}% "
            f"({best_errors}/{reference_words}) weights="
            f"{best_weights.tolist()}"
        )

    best_scores = base_scores + np.einsum(
        "ukf,f->uk",
        features,
        best_weights,
    )
    selected = best_scores.argmax(axis=1)
    predictions = [
        normalized_texts[row_index][candidate_index]
        for row_index, candidate_index in enumerate(selected)
    ]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "best.jsonl").open("w", encoding="utf-8") as handle:
        for row, prediction in zip(rows, predictions):
            handle.write(
                json.dumps(
                    {
                        "utterance_id": row["utterance_id"],
                        "reference": row["reference"],
                        "prediction": prediction,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    summary = {
        "candidates": args.candidates,
        "drop_words": sorted(dropped_words),
        "num_utterances": len(rows),
        "top_k": args.top_k,
        "samples_per_round": args.samples,
        "rounds": args.rounds,
        "seed": args.seed,
        "weight_order": WEIGHT_NAMES,
        "bounds": dict(zip(WEIGHT_NAMES, bounds.tolist())),
        "round_results": round_results,
        "best": round_results[-1],
    }
    summary_path = output_dir / "results.json"
    summary_path.write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {summary_path}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("candidates")
    parser.add_argument("output_dir")
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--drop-word", action="append", default=[])
    parser.add_argument("--drop-filled-pauses", action="store_true")
    parser.add_argument("--samples", type=int, default=50000)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--search-batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--refinement-fraction", type=float, default=0.2)
    parser.add_argument(
        "--transformer-range",
        type=parse_float_range,
        default=(0.05, 0.35),
    )
    parser.add_argument(
        "--word-bonus-range",
        type=parse_float_range,
        default=(0.5, 2.2),
    )
    parser.add_argument(
        "--forward-lm-range",
        type=parse_float_range,
        default=(-0.05, 0.08),
    )
    parser.add_argument(
        "--reverse-lm-range",
        type=parse_float_range,
        default=(-0.05, 0.08),
    )
    parser.add_argument(
        "--drop-penalty-range",
        type=parse_float_range,
        default=(-1.5, 1.5),
    )
    parser.add_argument("--anchor", action="append", type=parse_weights, default=[])
    return parser.parse_args()


def main():
    args = parse_args()
    if args.samples <= 0 or args.rounds <= 0:
        raise ValueError("samples and rounds must be positive")
    if not 0.0 < args.refinement_fraction <= 1.0:
        raise ValueError("refinement fraction must be in (0, 1]")
    search(args)


if __name__ == "__main__":
    main()
