#!/usr/bin/env python3
"""Train and apply a compact VoxMM subword LM for N-best rescoring."""

import argparse
import csv
import json
import math
import random
import re
from pathlib import Path

import jiwer
import sentencepiece as spm
import torch
import torch.nn as nn
import torch.nn.functional as F


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TOKENIZER = REPO_ROOT / "external/AVEC/datasets/LRS3/tokenizerbpe256.model"


def normalize_text(text):
    text = text.lower()
    text = re.sub(r"\((?:[^()/]+/)+([^()]+)\)", r" \1 ", text)
    text = re.sub(r"[^a-z0-9']+", " ", text)
    return " ".join(text.split())


class SubwordLSTMLM(nn.Module):
    def __init__(
        self,
        vocab_size,
        embedding_dim=256,
        hidden_dim=256,
        num_layers=2,
        dropout=0.2,
    ):
        super().__init__()
        if embedding_dim != hidden_dim:
            raise ValueError("Tied embeddings require embedding_dim == hidden_dim")
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.projection = nn.Linear(hidden_dim, vocab_size, bias=False)
        self.projection.weight = self.embedding.weight

    def forward(self, tokens):
        encoded = self.embedding(tokens)
        encoded, _ = self.lstm(encoded)
        return self.projection(self.dropout(encoded))


def read_sequences(manifest_path, tokenizer):
    sequences = []
    with Path(manifest_path).open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            text = normalize_text(row["reference"])
            tokens = tokenizer.encode(text)
            if tokens:
                sequences.append(tokens)
    return sequences


def make_batches(sequences, batch_size, shuffle, rng):
    indices = sorted(range(len(sequences)), key=lambda index: len(sequences[index]))
    batches = [
        indices[start : start + batch_size]
        for start in range(0, len(indices), batch_size)
    ]
    if shuffle:
        rng.shuffle(batches)
    for batch_indices in batches:
        yield [sequences[index] for index in batch_indices]


def collate_sequences(sequences, bos_id, eos_id, device):
    max_length = max(len(sequence) for sequence in sequences) + 1
    inputs = torch.zeros(
        (len(sequences), max_length),
        dtype=torch.long,
        device=device,
    )
    targets = torch.full(
        (len(sequences), max_length),
        -100,
        dtype=torch.long,
        device=device,
    )
    for index, sequence in enumerate(sequences):
        source = [bos_id] + sequence
        target = sequence + [eos_id]
        inputs[index, : len(source)] = torch.tensor(source, device=device)
        targets[index, : len(target)] = torch.tensor(target, device=device)
    return inputs, targets


def run_epoch(
    model,
    sequences,
    batch_size,
    bos_id,
    eos_id,
    device,
    optimizer=None,
    rng=None,
):
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_tokens = 0
    batches = make_batches(
        sequences,
        batch_size=batch_size,
        shuffle=training,
        rng=rng,
    )
    context = torch.enable_grad() if training else torch.inference_mode()
    with context:
        for batch in batches:
            inputs, targets = collate_sequences(batch, bos_id, eos_id, device)
            logits = model(inputs)
            loss_sum = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                targets.reshape(-1),
                ignore_index=-100,
                reduction="sum",
            )
            token_count = int((targets != -100).sum())
            if training:
                optimizer.zero_grad(set_to_none=True)
                (loss_sum / token_count).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            total_loss += float(loss_sum.detach())
            total_tokens += token_count
    return total_loss / max(total_tokens, 1)


def train(args):
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    tokenizer = spm.SentencePieceProcessor(args.tokenizer)
    sequences = read_sequences(args.manifest, tokenizer)
    if args.reverse:
        sequences = [list(reversed(sequence)) for sequence in sequences]
    rng.shuffle(sequences)

    validation_count = max(1, round(len(sequences) * args.validation_fraction))
    validation_sequences = sequences[:validation_count]
    training_sequences = sequences[validation_count:]
    base_vocab_size = tokenizer.vocab_size()
    bos_id = base_vocab_size
    eos_id = base_vocab_size + 1
    model_config = {
        "vocab_size": base_vocab_size + 2,
        "embedding_dim": args.embedding_dim,
        "hidden_dim": args.hidden_dim,
        "num_layers": args.num_layers,
        "dropout": args.dropout,
        "base_vocab_size": base_vocab_size,
        "bos_id": bos_id,
        "eos_id": eos_id,
        "reverse": args.reverse,
    }

    device = torch.device(
        args.device
        if args.device
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    model = SubwordLSTMLM(
        vocab_size=model_config["vocab_size"],
        embedding_dim=model_config["embedding_dim"],
        hidden_dim=model_config["hidden_dim"],
        num_layers=model_config["num_layers"],
        dropout=model_config["dropout"],
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    print(
        f"Training on {len(training_sequences):,} sentences; "
        f"validating on {len(validation_sequences):,}; device={device}"
    )
    best_validation_nll = float("inf")
    stale_epochs = 0
    checkpoint_path = Path(args.checkpoint)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, args.epochs + 1):
        training_nll = run_epoch(
            model,
            training_sequences,
            args.batch_size,
            bos_id,
            eos_id,
            device,
            optimizer=optimizer,
            rng=rng,
        )
        validation_nll = run_epoch(
            model,
            validation_sequences,
            args.batch_size,
            bos_id,
            eos_id,
            device,
        )
        print(
            f"epoch={epoch:02d} train_nll={training_nll:.4f} "
            f"train_ppl={math.exp(training_nll):.2f} "
            f"val_nll={validation_nll:.4f} "
            f"val_ppl={math.exp(validation_nll):.2f}"
        )
        if validation_nll < best_validation_nll:
            best_validation_nll = validation_nll
            stale_epochs = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "model_config": model_config,
                    "tokenizer": str(Path(args.tokenizer).resolve()),
                    "best_validation_nll": best_validation_nll,
                    "epoch": epoch,
                },
                checkpoint_path,
            )
        else:
            stale_epochs += 1
            if stale_epochs >= args.patience:
                print(f"Stopping after {stale_epochs} stale epoch(s)")
                break
    print(f"Wrote {checkpoint_path}")


def load_model(checkpoint_path, device):
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )
    config = checkpoint["model_config"]
    model = SubwordLSTMLM(
        vocab_size=config["vocab_size"],
        embedding_dim=config["embedding_dim"],
        hidden_dim=config["hidden_dim"],
        num_layers=config["num_layers"],
        dropout=config["dropout"],
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, config, checkpoint


def load_candidate_rows(path):
    rows = []
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            rows.append(json.loads(line))
    return rows


def score_candidate_sequences(
    model,
    sequences,
    bos_id,
    eos_id,
    batch_size,
    device,
):
    scores = []
    with torch.inference_mode():
        for start in range(0, len(sequences), batch_size):
            batch = sequences[start : start + batch_size]
            inputs, targets = collate_sequences(batch, bos_id, eos_id, device)
            log_probs = model(inputs).log_softmax(dim=-1)
            target_tokens = targets.clamp_min(0)
            token_scores = log_probs.gather(
                dim=-1,
                index=target_tokens.unsqueeze(-1),
            ).squeeze(-1)
            token_scores = token_scores.masked_fill(targets == -100, 0.0)
            scores.extend(token_scores.sum(dim=-1).cpu().tolist())
    return scores


def parse_rescore_setting(value):
    fields = value.split(":")
    if len(fields) not in {3, 4, 5}:
        raise argparse.ArgumentTypeError(
            "setting must be "
            "name:neural_lm_weight:word_bonus[:base_weight[:aux_lm_weight]]"
        )
    return {
        "name": fields[0],
        "neural_lm_weight": float(fields[1]),
        "word_bonus": float(fields[2]),
        "base_weight": float(fields[3]) if len(fields) >= 4 else 1.0,
        "aux_lm_weight": float(fields[4]) if len(fields) == 5 else 0.0,
    }


def error_counts(references, predictions):
    result = jiwer.process_words(references, predictions)
    return {
        "hits": result.hits,
        "substitutions": result.substitutions,
        "deletions": result.deletions,
        "insertions": result.insertions,
    }


def rescore(args):
    device = torch.device(
        args.device
        if args.device
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    model, config, checkpoint = load_model(args.checkpoint, device)
    rows = load_candidate_rows(args.candidates)
    flattened = []
    for row in rows:
        for candidate in row["candidates"]:
            flattened.append(candidate)

    sequences = [candidate["tokens"] for candidate in flattened]
    if config.get("reverse", False):
        sequences = [list(reversed(sequence)) for sequence in sequences]
    scores = score_candidate_sequences(
        model,
        sequences,
        config["bos_id"],
        config["eos_id"],
        args.batch_size,
        device,
    )
    for candidate, score in zip(flattened, scores):
        candidate["neural_lm_score"] = score
        candidate["word_count"] = len(candidate["text"].split())

    aux_checkpoint = None
    if args.aux_checkpoint:
        aux_model, aux_config, aux_checkpoint = load_model(
            args.aux_checkpoint,
            device,
        )
        aux_sequences = [candidate["tokens"] for candidate in flattened]
        if aux_config.get("reverse", False):
            aux_sequences = [
                list(reversed(sequence)) for sequence in aux_sequences
            ]
        aux_scores = score_candidate_sequences(
            aux_model,
            aux_sequences,
            aux_config["bos_id"],
            aux_config["eos_id"],
            args.batch_size,
            device,
        )
        for candidate, score in zip(flattened, aux_scores):
            candidate["aux_neural_lm_score"] = score

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    scored_path = output_dir / "scored_nbest.jsonl"
    with scored_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(
        f"Scored {len(flattened):,} candidates with checkpoint epoch "
        f"{checkpoint['epoch']} on {device}"
    )
    if aux_checkpoint is not None:
        print(
            "Scored candidates with auxiliary checkpoint epoch "
            f"{aux_checkpoint['epoch']}"
        )
    print(f"Wrote {scored_path}")

    references = [row["reference"] for row in rows]
    settings = args.setting or [
        parse_rescore_setting("n010_b010:0.10:0.10"),
        parse_rescore_setting("n020_b020:0.20:0.20"),
        parse_rescore_setting("n030_b030:0.30:0.30"),
    ]
    results = []
    for setting in settings:
        predictions = []
        for row in rows:
            candidate = max(
                row["candidates"],
                key=lambda item: (
                    setting["base_weight"] * item["score"]
                    + setting["neural_lm_weight"] * item["neural_lm_score"]
                    + setting["aux_lm_weight"]
                    * item.get("aux_neural_lm_score", 0.0)
                    + setting["word_bonus"] * item["word_count"]
                ),
            )
            predictions.append(candidate["text"])

        wer = 100.0 * jiwer.wer(references, predictions)
        result = {
            **setting,
            "wer": wer,
            **error_counts(references, predictions),
        }
        results.append(result)
        print(
            f"{setting['name']}: WER={wer:.4f}% "
            f"(S={result['substitutions']}, D={result['deletions']}, "
            f"I={result['insertions']})"
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
        "checkpoint": args.checkpoint,
        "checkpoint_epoch": checkpoint["epoch"],
        "aux_checkpoint": args.aux_checkpoint,
        "aux_checkpoint_epoch": (
            aux_checkpoint["epoch"] if aux_checkpoint is not None else None
        ),
        "num_utterances": len(rows),
        "num_candidates": len(flattened),
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

    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("manifest")
    train_parser.add_argument("checkpoint")
    train_parser.add_argument("--tokenizer", default=str(DEFAULT_TOKENIZER))
    train_parser.add_argument("--epochs", type=int, default=15)
    train_parser.add_argument("--patience", type=int, default=3)
    train_parser.add_argument("--validation-fraction", type=float, default=0.05)
    train_parser.add_argument("--batch-size", type=int, default=128)
    train_parser.add_argument("--embedding-dim", type=int, default=256)
    train_parser.add_argument("--hidden-dim", type=int, default=256)
    train_parser.add_argument("--num-layers", type=int, default=2)
    train_parser.add_argument("--dropout", type=float, default=0.2)
    train_parser.add_argument("--learning-rate", type=float, default=2e-3)
    train_parser.add_argument("--weight-decay", type=float, default=1e-4)
    train_parser.add_argument("--seed", type=int, default=7)
    train_parser.add_argument("--reverse", action="store_true")
    train_parser.add_argument("--device")
    train_parser.set_defaults(func=train)

    rescore_parser = subparsers.add_parser("rescore")
    rescore_parser.add_argument("candidates")
    rescore_parser.add_argument("checkpoint")
    rescore_parser.add_argument("output_dir")
    rescore_parser.add_argument("--aux-checkpoint")
    rescore_parser.add_argument("--batch-size", type=int, default=256)
    rescore_parser.add_argument("--device")
    rescore_parser.add_argument(
        "--setting",
        action="append",
        type=parse_rescore_setting,
    )
    rescore_parser.set_defaults(func=rescore)

    return parser.parse_args()


def main():
    args = parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
