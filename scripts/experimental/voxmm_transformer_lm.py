#!/usr/bin/env python3
"""Score and rerank VoxMM N-best hypotheses with neural language models."""

import argparse
import json
import math
import time
from pathlib import Path


DEFAULT_MODEL = "distilbert/distilgpt2"
DEFAULT_MASKED_MODEL = "FacebookAI/roberta-base"
DEFAULT_SPEECHBRAIN_DIR = (
    Path(__file__).resolve().parents[2]
    / "external/assets/lm/speechbrain-librispeech-transformer"
)
FILLED_PAUSE_WORDS = frozenset(
    {
        "aah",
        "ah",
        "aha",
        "ahh",
        "buh",
        "eh",
        "er",
        "erm",
        "hm",
        "hmm",
        "huh",
        "mhm",
        "mm",
        "mmh",
        "nuh",
        "ruh",
        "suh",
        "uh",
        "uhh",
        "um",
        "umm",
    }
)


def load_rows(path):
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def format_text(text, style):
    text = text.strip()
    if style == "raw" or not text:
        return text
    if style == "sentence":
        return text[0].upper() + text[1:] + "."
    raise ValueError(f"Unsupported text style: {style}")


def dropped_words_from_args(args):
    words = {word.lower() for word in args.drop_word}
    if args.drop_filled_pauses:
        words.update(FILLED_PAUSE_WORDS)
    return words


def score_masked_lm(args, device):
    import torch
    import torch.nn.functional as F
    from transformers import AutoModelForMaskedLM, AutoTokenizer

    rows = load_rows(args.candidates)
    dropped_words = dropped_words_from_args(args)
    tokenizer = AutoTokenizer.from_pretrained(args.masked_model)
    model = AutoModelForMaskedLM.from_pretrained(args.masked_model).to(device)
    model.eval()

    def candidate_text(candidate):
        normalized, _ = normalize_hypothesis(candidate["text"], dropped_words)
        return format_text(normalized, args.text_style)

    unique_texts = []
    seen = set()
    for row in rows:
        row["candidates"] = row["candidates"][: args.top_k]
        for candidate in row["candidates"]:
            text = candidate_text(candidate)
            if text not in seen:
                seen.add(text)
                unique_texts.append(text)

    sequences = {
        text: tokenizer.encode(text, add_special_tokens=True)
        for text in unique_texts
    }
    special_ids = set(tokenizer.all_special_ids)
    tasks = []
    for text, token_ids in sequences.items():
        for position, token_id in enumerate(token_ids):
            if token_id not in special_ids:
                tasks.append((text, position, token_id))
    tasks.sort(key=lambda item: len(sequences[item[0]]))
    scores = {text: 0.0 for text in unique_texts}
    started = time.perf_counter()
    batch_count = math.ceil(len(tasks) / args.batch_size)

    with torch.inference_mode():
        for batch_index, start in enumerate(
            range(0, len(tasks), args.batch_size),
            start=1,
        ):
            batch = tasks[start : start + args.batch_size]
            max_length = max(len(sequences[text]) for text, _, _ in batch)
            input_ids = torch.full(
                (len(batch), max_length),
                tokenizer.pad_token_id,
                dtype=torch.long,
                device=device,
            )
            attention_mask = torch.zeros_like(input_ids)
            positions = []
            targets = []
            for row_index, (text, position, target) in enumerate(batch):
                token_ids = sequences[text]
                length = len(token_ids)
                input_ids[row_index, :length] = torch.tensor(
                    token_ids,
                    dtype=torch.long,
                    device=device,
                )
                input_ids[row_index, position] = tokenizer.mask_token_id
                attention_mask[row_index, :length] = 1
                positions.append(position)
                targets.append(target)

            row_indices = torch.arange(len(batch), device=device)
            position_indices = torch.tensor(positions, device=device)
            if hasattr(model, "roberta") and hasattr(model, "lm_head"):
                hidden = model.roberta(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                ).last_hidden_state
                masked_hidden = hidden[row_indices, position_indices]
                logits = model.lm_head(masked_hidden)
            else:
                all_logits = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                ).logits
                logits = all_logits[row_indices, position_indices]
            target_tensor = torch.tensor(targets, device=device)
            token_scores = -F.cross_entropy(
                logits,
                target_tensor,
                reduction="none",
            )
            for (text, _, _), value in zip(
                batch,
                token_scores.cpu().tolist(),
            ):
                scores[text] += float(value)

            if batch_index == 1 or batch_index % args.progress_every == 0:
                elapsed = time.perf_counter() - started
                print(
                    f"Scored MLM batch {batch_index:,}/{batch_count:,} "
                    f"({min(start + len(batch), len(tasks)):,}/"
                    f"{len(tasks):,} masked tokens, {elapsed:.1f}s)",
                    flush=True,
                )

    for row in rows:
        for candidate in row["candidates"]:
            candidate["transformer_lm_score"] = scores[candidate_text(candidate)]

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(
        f"Wrote {output_path} with {len(unique_texts):,} unique "
        "masked-LM scores"
    )


def load_scorer(args, device):
    import torch

    if args.backend == "hf":
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(args.model)
        if tokenizer.eos_token_id is None:
            raise ValueError("The causal LM tokenizer must define an EOS token.")
        model = AutoModelForCausalLM.from_pretrained(args.model).to(device)
        model.eval()
        model.config.use_cache = False

        def encode(text):
            return tokenizer.encode(text, add_special_tokens=False)

        def model_logits(input_ids, attention_mask):
            return model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            ).logits

        token_id = tokenizer.eos_token_id
        return encode, model_logits, token_id, token_id, token_id

    import sentencepiece as spm
    from speechbrain.lobes.models.transformer.TransformerLM import TransformerLM

    model_dir = Path(args.speechbrain_dir)
    tokenizer = spm.SentencePieceProcessor(
        model_file=str(model_dir / "tokenizer.ckpt")
    )
    model = TransformerLM(
        vocab=5000,
        d_model=768,
        nhead=12,
        num_encoder_layers=12,
        num_decoder_layers=0,
        d_ffn=3072,
        dropout=0.0,
        activation=torch.nn.GELU,
        normalize_before=False,
    ).to(device)
    model.load_state_dict(
        torch.load(model_dir / "lm.ckpt", map_location=device)
    )
    model.eval()

    def encode(text):
        return tokenizer.encode(text.upper(), out_type=int)

    def model_logits(input_ids, attention_mask):
        del attention_mask
        return model(input_ids)

    return encode, model_logits, 1, 2, 0


def score(args):
    import torch
    import torch.nn.functional as F

    device = torch.device(
        args.device
        if args.device
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    if args.backend == "mlm":
        return score_masked_lm(args, device)
    encode, model_logits, bos_id, eos_id, pad_id = load_scorer(args, device)

    rows = load_rows(args.candidates)
    formatted_texts = []
    seen = set()
    for row in rows:
        for candidate in row["candidates"]:
            text = format_text(candidate["text"], args.text_style)
            if text not in seen:
                seen.add(text)
                formatted_texts.append(text)

    encoded = {text: encode(text) for text in formatted_texts}
    ordered_texts = sorted(formatted_texts, key=lambda text: len(encoded[text]))
    scores = {}
    started = time.perf_counter()
    batch_count = math.ceil(len(ordered_texts) / args.batch_size)

    with torch.inference_mode():
        for batch_index, start in enumerate(
            range(0, len(ordered_texts), args.batch_size),
            start=1,
        ):
            texts = ordered_texts[start : start + args.batch_size]
            sequences = [[bos_id, *encoded[text], eos_id] for text in texts]
            max_length = max(len(sequence) for sequence in sequences)
            input_ids = torch.full(
                (len(sequences), max_length),
                pad_id,
                dtype=torch.long,
                device=device,
            )
            attention_mask = torch.zeros_like(input_ids)
            for row_index, sequence in enumerate(sequences):
                length = len(sequence)
                input_ids[row_index, :length] = torch.tensor(
                    sequence,
                    dtype=torch.long,
                    device=device,
                )
                attention_mask[row_index, :length] = 1

            logits = model_logits(input_ids, attention_mask)[:, :-1]
            targets = input_ids[:, 1:]
            target_mask = attention_mask[:, 1:]
            losses = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                targets.reshape(-1),
                reduction="none",
            ).reshape_as(targets)
            batch_scores = -(losses * target_mask).sum(dim=1)
            scores.update(
                (text, float(value))
                for text, value in zip(texts, batch_scores.cpu().tolist())
            )

            if batch_index == 1 or batch_index % args.progress_every == 0:
                elapsed = time.perf_counter() - started
                print(
                    f"Scored batch {batch_index:,}/{batch_count:,} "
                    f"({min(start + len(texts), len(ordered_texts)):,}/"
                    f"{len(ordered_texts):,} unique texts, {elapsed:.1f}s)"
                )

    for row in rows:
        for candidate in row["candidates"]:
            text = format_text(candidate["text"], args.text_style)
            candidate["transformer_lm_score"] = scores[text]

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(
        f"Wrote {output_path} with {len(formatted_texts):,} unique "
        f"transformer scores"
    )


def parse_setting(value):
    fields = value.split(":")
    if len(fields) not in {3, 4, 5, 6, 7, 8}:
        raise argparse.ArgumentTypeError(
            "setting must be name:transformer_weight:word_bonus"
            "[:base_weight[:drop_penalty[:posterior_temperature"
            "[:forward_lm_weight[:reverse_lm_weight]]]]]"
        )
    return {
        "name": fields[0],
        "transformer_weight": float(fields[1]),
        "word_bonus": float(fields[2]),
        "base_weight": float(fields[3]) if len(fields) >= 4 else 1.0,
        "drop_penalty": float(fields[4]) if len(fields) >= 5 else 0.0,
        "posterior_temperature": float(fields[5]) if len(fields) >= 6 else 0.0,
        "forward_lm_weight": float(fields[6]) if len(fields) >= 7 else 0.0,
        "reverse_lm_weight": float(fields[7]) if len(fields) == 8 else 0.0,
    }


def normalize_hypothesis(text, dropped_words):
    words = text.split()
    kept = [word for word in words if word.lower() not in dropped_words]
    return " ".join(kept), len(words) - len(kept)


def combined_score(candidate, setting, dropped_words):
    normalized, dropped_count = normalize_hypothesis(
        candidate["text"],
        dropped_words,
    )
    return (
        setting["base_weight"] * candidate["score"]
        + setting["transformer_weight"] * candidate["transformer_lm_score"]
        + setting["forward_lm_weight"] * candidate.get("neural_lm_score", 0.0)
        + setting["reverse_lm_weight"]
        * candidate.get("aux_neural_lm_score", 0.0)
        + setting["word_bonus"] * len(normalized.split())
        - setting["drop_penalty"] * dropped_count
    ), normalized


def select_prediction(candidates, setting, dropped_words):
    scored = [
        (*combined_score(candidate, setting, dropped_words), candidate)
        for candidate in candidates
    ]
    temperature = setting["posterior_temperature"]
    if temperature <= 0.0:
        return max(scored, key=lambda item: item[0])[1]

    maximum = max(item[0] for item in scored)
    mass_by_text = {}
    for value, text, _ in scored:
        mass_by_text[text] = mass_by_text.get(text, 0.0) + math.exp(
            (value - maximum) / temperature
        )
    return max(mass_by_text.items(), key=lambda item: item[1])[0]


def word_errors(reference, hypothesis):
    reference_words = reference.split()
    hypothesis_words = hypothesis.split()
    previous = list(range(len(hypothesis_words) + 1))
    for reference_index, reference_word in enumerate(reference_words, start=1):
        current = [reference_index]
        for hypothesis_index, hypothesis_word in enumerate(
            hypothesis_words,
            start=1,
        ):
            current.append(
                min(
                    current[-1] + 1,
                    previous[hypothesis_index] + 1,
                    previous[hypothesis_index - 1]
                    + (reference_word != hypothesis_word),
                )
            )
        previous = current
    return previous[-1], len(reference_words)


def rescore(args):
    rows = load_rows(args.candidates)
    dropped_words = dropped_words_from_args(args)
    settings = args.setting or [
        parse_setting("t001_b000:0.01:0.0"),
        parse_setting("t003_b010:0.03:0.1"),
        parse_setting("t005_b020:0.05:0.2"),
    ]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results = []

    for setting in settings:
        predictions = [
            select_prediction(row["candidates"], setting, dropped_words)
            for row in rows
        ]
        errors = 0
        reference_words = 0
        for row, prediction in zip(rows, predictions):
            utterance_errors, utterance_words = word_errors(
                row["reference"],
                prediction,
            )
            errors += utterance_errors
            reference_words += utterance_words
        wer = 100.0 * errors / max(reference_words, 1)
        result = {
            **setting,
            "wer": wer,
            "errors": errors,
            "reference_words": reference_words,
        }
        results.append(result)
        print(
            f"{setting['name']}: WER={wer:.4f}% "
            f"({errors}/{reference_words} word errors)"
        )

        prediction_path = output_dir / f"{setting['name']}.jsonl"
        with prediction_path.open("w", encoding="utf-8") as handle:
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
        "results": sorted(results, key=lambda item: item["wer"]),
    }
    summary_path = output_dir / "results.json"
    summary_path.write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {summary_path}")


def parse_args():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    score_parser = subparsers.add_parser("score")
    score_parser.add_argument("candidates")
    score_parser.add_argument("output")
    score_parser.add_argument(
        "--backend",
        choices=("hf", "speechbrain", "mlm"),
        default="hf",
    )
    score_parser.add_argument("--model", default=DEFAULT_MODEL)
    score_parser.add_argument("--masked-model", default=DEFAULT_MASKED_MODEL)
    score_parser.add_argument(
        "--speechbrain-dir",
        default=str(DEFAULT_SPEECHBRAIN_DIR),
    )
    score_parser.add_argument(
        "--text-style",
        choices=("raw", "sentence"),
        default="raw",
    )
    score_parser.add_argument("--top-k", type=int, default=20)
    score_parser.add_argument("--drop-word", action="append", default=[])
    score_parser.add_argument("--drop-filled-pauses", action="store_true")
    score_parser.add_argument("--batch-size", type=int, default=16)
    score_parser.add_argument("--progress-every", type=int, default=100)
    score_parser.add_argument("--device")
    score_parser.set_defaults(func=score)

    rescore_parser = subparsers.add_parser("rescore")
    rescore_parser.add_argument("candidates")
    rescore_parser.add_argument("output_dir")
    rescore_parser.add_argument("--drop-word", action="append", default=[])
    rescore_parser.add_argument("--drop-filled-pauses", action="store_true")
    rescore_parser.add_argument(
        "--setting",
        action="append",
        type=parse_setting,
    )
    rescore_parser.set_defaults(func=rescore)

    return parser.parse_args()


def main():
    args = parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
