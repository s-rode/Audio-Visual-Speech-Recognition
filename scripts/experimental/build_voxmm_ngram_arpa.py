#!/usr/bin/env python3
"""Build a compact backoff ARPA LM directly from a VoxMM manifest."""

import argparse
import csv
import math
import re
from collections import Counter
from functools import lru_cache
from pathlib import Path


def words(text):
    text = text.lower()
    text = re.sub(r"\((?:[^()/]+/)+([^()]+)\)", r" \1 ", text)
    text = re.sub(r"[^a-z0-9']+", " ", text)
    return [word for word in text.split() if word]


class AbsoluteDiscountLM:
    def __init__(
        self, order=4, discount=0.75, floor=1e-12, mode="interpolated"
    ):
        self.order = max(1, int(order))
        self.discount = float(discount)
        self.floor = float(floor)
        if mode not in {"interpolated", "backoff"}:
            raise ValueError(f"Unsupported ARPA mode: {mode}")
        self.mode = mode
        self.ngram_counts = [Counter() for _ in range(self.order + 1)]
        self.context_counts = [Counter() for _ in range(self.order + 1)]
        self.follower_counts = [Counter() for _ in range(self.order + 1)]
        self.followers = [dict() for _ in range(self.order + 1)]
        self.continuation_counts = Counter()
        self.total_continuations = 0
        self.vocab = {"<unk>", "</s>"}

    def fit(self, sentences):
        sentences = list(sentences)
        for sentence in sentences:
            self.vocab.update(sentence)

        followers = [dict() for _ in range(self.order + 1)]
        for sentence in sentences:
            tokens = ["<s>"]
            tokens.extend(sentence)
            tokens.append("</s>")
            for index in range(1, len(tokens)):
                token = tokens[index]
                for ngram_order in range(1, self.order + 1):
                    start = index - ngram_order + 1
                    if start < 0:
                        continue
                    context = tuple(tokens[start:index])
                    ngram = context + (token,)
                    self.ngram_counts[ngram_order][ngram] += 1
                    self.context_counts[ngram_order][context] += 1
                    if ngram_order > 1:
                        followers[ngram_order].setdefault(context, set()).add(token)

        self.followers = followers
        for ngram_order in range(2, self.order + 1):
            self.follower_counts[ngram_order] = Counter(
                {context: len(tokens) for context, tokens in followers[ngram_order].items()}
            )

        predecessors = {}
        if self.order >= 2:
            for predecessor, token in self.ngram_counts[2]:
                predecessors.setdefault(token, set()).add(predecessor)
        self.continuation_counts = Counter(
            {token: len(values) for token, values in predecessors.items()}
        )
        self.total_continuations = sum(self.continuation_counts.values())
        self.probability.cache_clear()
        self.backoff_weight.cache_clear()
        return self

    @lru_cache(maxsize=None)
    def probability(self, ngram):
        ngram_order = len(ngram)
        if ngram_order == 1:
            return self._unigram_probability(ngram[0])

        history = ngram[:-1]
        context_count = self.context_counts[ngram_order][history]
        ngram_count = self.ngram_counts[ngram_order][ngram]
        lower_probability = self.probability(ngram[1:])
        if context_count == 0:
            return lower_probability

        discounted = max(ngram_count - self.discount, 0.0) / context_count
        if self.mode == "interpolated":
            return discounted + self.backoff_weight(history) * lower_probability
        if ngram_count > 0:
            return max(discounted, self.floor)
        return self.backoff_weight(history) * lower_probability

    @lru_cache(maxsize=None)
    def backoff_weight(self, context):
        next_order = len(context) + 1
        if next_order > self.order:
            return 1.0
        context_count = self.context_counts[next_order][context]
        if context_count == 0:
            return 1.0
        if self.mode == "interpolated":
            followers = self.follower_counts[next_order][context]
            return self.discount * followers / context_count

        follower_tokens = self.followers[next_order].get(context, ())
        discounted_seen_mass = sum(
            max(self.ngram_counts[next_order][context + (token,)] - self.discount, 0.0)
            for token in follower_tokens
        ) / context_count
        remaining_mass = max(1.0 - discounted_seen_mass, self.floor)
        lower_seen_mass = sum(
            self.probability(context[1:] + (token,))
            for token in follower_tokens
        )
        lower_unseen_mass = max(1.0 - lower_seen_mass, self.floor)
        return max(remaining_mass / lower_unseen_mass, self.floor)

    def _unigram_probability(self, token):
        if token == "<s>":
            return self.floor
        if self.total_continuations <= 0:
            return 1.0 / max(len(self.vocab), 1)
        numerator = self.continuation_counts[token] + self.floor
        denominator = self.total_continuations + self.floor * len(self.vocab)
        return numerator / denominator

    def iter_ngrams(self, ngram_order):
        ngrams = set(self.ngram_counts[ngram_order])
        if ngram_order == 1:
            ngrams.update({("<s>",), ("<unk>",), ("</s>",)})
        return sorted(ngrams)

    def write_arpa(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        ngrams_by_order = {
            ngram_order: self.iter_ngrams(ngram_order)
            for ngram_order in range(1, self.order + 1)
        }

        with path.open("w", encoding="utf-8") as handle:
            handle.write("\\data\\\n")
            for ngram_order, ngrams in ngrams_by_order.items():
                handle.write(f"ngram {ngram_order}={len(ngrams)}\n")

            for ngram_order, ngrams in ngrams_by_order.items():
                handle.write(f"\n\\{ngram_order}-grams:\n")
                for ngram in ngrams:
                    probability = max(self.probability(ngram), self.floor)
                    fields = [f"{math.log10(probability):.7f}", " ".join(ngram)]
                    if ngram_order < self.order:
                        backoff = max(self.backoff_weight(ngram), self.floor)
                        fields.append(f"{math.log10(backoff):.7f}")
                    handle.write("\t".join(fields) + "\n")
            handle.write("\n\\end\\\n")


def read_sentences(manifest_path):
    with Path(manifest_path).open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            sentence = words(row["reference"])
            if sentence:
                yield sentence


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", help="Training manifest containing a reference column")
    parser.add_argument("output", help="Output ARPA path")
    parser.add_argument("--order", type=int, default=4)
    parser.add_argument("--discount", type=float, default=0.75)
    parser.add_argument(
        "--mode",
        choices=["interpolated", "backoff"],
        default="interpolated",
    )
    parser.add_argument("--floor", type=float, default=1e-12)
    return parser.parse_args()


def main():
    args = parse_args()
    lm = AbsoluteDiscountLM(
        order=args.order,
        discount=args.discount,
        floor=args.floor,
        mode=args.mode,
    ).fit(read_sentences(args.manifest))
    lm.write_arpa(args.output)
    counts = [len(lm.iter_ngrams(order)) for order in range(1, lm.order + 1)]
    print(f"Wrote {args.output}")
    print("N-grams:", ", ".join(f"{order}={count:,}" for order, count in enumerate(counts, 1)))


if __name__ == "__main__":
    main()
