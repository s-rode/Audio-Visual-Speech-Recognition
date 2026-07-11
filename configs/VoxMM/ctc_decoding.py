import csv
import math
import os
import re
import warnings
from collections import Counter

import sentencepiece as spm
import torch
import torch.nn as nn


NEG_INF = -float("inf")


def _logadd(*values):
    finite = [value for value in values if value != NEG_INF]
    if not finite:
        return NEG_INF
    max_value = max(finite)
    return max_value + math.log(sum(math.exp(value - max_value) for value in finite))


class SimpleWordNGramLM:
    def __init__(self, manifest_path, order=3, smoothing=0.1):
        self.order = max(1, int(order))
        self.smoothing = float(smoothing)
        self.ngram_counts = [Counter() for _ in range(self.order + 1)]
        self.context_counts = [Counter() for _ in range(self.order + 1)]
        self.vocab = {"<unk>", "</s>"}

        self._fit(manifest_path)
        self.vocab_size = len(self.vocab)

    @staticmethod
    def words(text):
        text = text.lower()
        text = re.sub(r"[^a-z0-9']+", " ", text)
        return [word for word in text.split() if word]

    def score(self, text):
        return self.score_words(self.words(text))

    def score_words(self, words, add_eos=True):
        tokens = [word if word in self.vocab else "<unk>" for word in words]
        if add_eos:
            tokens.append("</s>")
        history = ["<s>"] * (self.order - 1)
        score = 0.0
        for token in tokens:
            score += self._score_next(history, token)
            history.append(token)
        return score

    def _fit(self, manifest_path):
        with open(manifest_path, newline="", encoding="utf-8") as handle:
            corpus = [self.words(row["reference"]) for row in csv.DictReader(handle)]

        for words in corpus:
            self.vocab.update(words)

        for words in corpus:
            tokens = ["<s>"] * (self.order - 1)
            tokens.extend(word if word in self.vocab else "<unk>" for word in words)
            tokens.append("</s>")
            for idx in range(self.order - 1, len(tokens)):
                token = tokens[idx]
                for order in range(1, self.order + 1):
                    start = idx - order + 1
                    if start < 0:
                        continue
                    context = tuple(tokens[start:idx])
                    ngram = context + (token,)
                    self.ngram_counts[order][ngram] += 1
                    self.context_counts[order][context] += 1

    def score_next(self, history, token):
        return self._score_next(list(history), token)

    def _score_next(self, history, token):
        token = token if token in self.vocab else "<unk>"
        order = min(self.order, len(history) + 1)
        context = tuple(history[-(order - 1):]) if order > 1 else ()
        ngram = context + (token,)
        numerator = self.ngram_counts[order][ngram] + self.smoothing
        denominator = self.context_counts[order][context] + self.smoothing * self.vocab_size
        return math.log(numerator / denominator)


class KneserNeyWordNGramLM:
    def __init__(self, manifest_path, order=4, discount=0.75, floor=1e-12):
        self.order = max(1, int(order))
        self.discount = float(discount)
        self.floor = float(floor)
        self.ngram_counts = [Counter() for _ in range(self.order + 1)]
        self.context_counts = [Counter() for _ in range(self.order + 1)]
        self.followers = [dict() for _ in range(self.order + 1)]
        self.continuation_counts = Counter()
        self.vocab = {"<unk>", "</s>"}
        self.total_continuations = 0

        self._fit(manifest_path)
        self.vocab_size = len(self.vocab)

    words = staticmethod(SimpleWordNGramLM.words)

    def score(self, text):
        return self.score_words(self.words(text))

    def score_words(self, words, add_eos=True):
        tokens = [word if word in self.vocab else "<unk>" for word in words]
        if add_eos:
            tokens.append("</s>")
        history = ["<s>"] * (self.order - 1)
        score = 0.0
        for token in tokens:
            score += self.score_next(history, token)
            history.append(token)
        return score

    def score_next(self, history, token):
        token = token if token in self.vocab else "<unk>"
        probability = self._prob(self.order, tuple(history), token)
        return math.log(max(probability, self.floor))

    def _fit(self, manifest_path):
        with open(manifest_path, newline="", encoding="utf-8") as handle:
            corpus = [self.words(row["reference"]) for row in csv.DictReader(handle)]

        for words in corpus:
            self.vocab.update(words)

        for words in corpus:
            tokens = ["<s>"] * (self.order - 1)
            tokens.extend(word if word in self.vocab else "<unk>" for word in words)
            tokens.append("</s>")
            for idx in range(self.order - 1, len(tokens)):
                token = tokens[idx]
                for order in range(1, self.order + 1):
                    start = idx - order + 1
                    if start < 0:
                        continue
                    context = tuple(tokens[start:idx])
                    ngram = context + (token,)
                    self.ngram_counts[order][ngram] += 1
                    self.context_counts[order][context] += 1
                    if order > 1:
                        self.followers[order].setdefault(context, set()).add(token)

        bigrams = self.ngram_counts[2] if self.order >= 2 else Counter()
        predecessors = {}
        for ngram in bigrams:
            predecessor, token = ngram
            predecessors.setdefault(token, set()).add(predecessor)
        self.continuation_counts = Counter({token: len(values) for token, values in predecessors.items()})
        self.total_continuations = sum(self.continuation_counts.values())

    def _prob(self, order, history, token):
        if order <= 1:
            return self._unigram_prob(token)

        context = tuple(history[-(order - 1):]) if order > 1 else ()
        context_count = self.context_counts[order][context]
        if context_count == 0:
            return self._prob(order - 1, history, token)

        ngram = context + (token,)
        ngram_count = self.ngram_counts[order][ngram]
        followers = len(self.followers[order].get(context, ()))
        lower = self._prob(order - 1, history, token)
        discounted = max(ngram_count - self.discount, 0.0) / context_count
        backoff_weight = self.discount * followers / context_count
        return discounted + backoff_weight * lower

    def _unigram_prob(self, token):
        if self.total_continuations <= 0:
            return 1.0 / max(self.vocab_size, 1)
        numerator = self.continuation_counts[token] + self.floor
        denominator = self.total_continuations + self.floor * self.vocab_size
        return numerator / denominator


class CTCPrefixBeamSearchDecoder(nn.Module):
    """Dependency-free CTC prefix beam search with optional final LM rescoring."""

    def __init__(
        self,
        tokenizer_path,
        beam_size=16,
        blank_token=0,
        cutoff_top_n=40,
        temperature=1.0,
        token_bonus=0.0,
        lm=None,
        lm_weight=0.0,
        word_bonus=0.0,
    ):
        super().__init__()
        self.tokenizer = spm.SentencePieceProcessor(tokenizer_path)
        self.beam_size = int(beam_size)
        self.blank_token = int(blank_token)
        self.cutoff_top_n = int(cutoff_top_n)
        self.temperature = float(temperature)
        self.token_bonus = float(token_bonus)
        self.lm = lm
        self.lm_weight = float(lm_weight)
        self.word_bonus = float(word_bonus)
        self._lm_score_cache = {}

    def forward(self, outputs, from_logits=True):
        if from_logits:
            tokens = self.beam_search(*outputs)
        else:
            tokens = outputs[0].tolist()
        return self.tokenizer.decode(tokens)

    def beam_search(self, logits, logits_len):
        logits = logits.detach().float().cpu() / self.temperature
        logits_len = logits_len.detach().cpu().tolist()
        log_probs = logits.log_softmax(dim=-1)

        results = []
        for batch_idx, length in enumerate(logits_len):
            results.append(self._decode_one(log_probs[batch_idx, : int(length)]))
        return results

    def _decode_one(self, log_probs):
        vocab_size = log_probs.shape[-1]
        cutoff = min(max(self.cutoff_top_n, 1), vocab_size)
        self._lm_score_cache = {}

        # prefix -> (prob ending in blank, prob ending in non-blank)
        beam = {(): (0.0, NEG_INF)}

        for timestep in range(log_probs.shape[0]):
            top_tokens = torch.topk(log_probs[timestep], k=cutoff).indices.tolist()
            if self.blank_token not in top_tokens:
                top_tokens.append(self.blank_token)

            next_beam = {}
            for prefix, (prob_blank, prob_non_blank) in beam.items():
                for token in top_tokens:
                    prob_token = float(log_probs[timestep, token])
                    old_blank, old_non_blank = next_beam.get(prefix, (NEG_INF, NEG_INF))

                    if token == self.blank_token:
                        old_blank = _logadd(old_blank, prob_blank + prob_token, prob_non_blank + prob_token)
                        next_beam[prefix] = (old_blank, old_non_blank)
                        continue

                    new_prefix = prefix + (token,)
                    new_blank, new_non_blank = next_beam.get(new_prefix, (NEG_INF, NEG_INF))

                    if prefix and token == prefix[-1]:
                        # Repeating the last non-blank token either keeps the same
                        # collapsed prefix, or creates a new prefix if separated by blank.
                        old_non_blank = _logadd(old_non_blank, prob_non_blank + prob_token)
                        new_non_blank = _logadd(new_non_blank, prob_blank + prob_token)
                        next_beam[prefix] = (old_blank, old_non_blank)
                        next_beam[new_prefix] = (new_blank, new_non_blank)
                    else:
                        new_non_blank = _logadd(new_non_blank, prob_blank + prob_token, prob_non_blank + prob_token)
                        next_beam[new_prefix] = (new_blank, new_non_blank)

            beam = dict(
                sorted(
                    next_beam.items(),
                    key=lambda item: self._prefix_score(item[0], item[1]),
                    reverse=True,
                )[: self.beam_size]
            )

        best_prefix = max(beam.items(), key=lambda item: self._final_score(item[0], item[1]))[0]
        return list(best_prefix)

    def _prefix_score(self, prefix, probs):
        score = _logadd(*probs) + self.token_bonus * len(prefix)
        return score + self._lm_prefix_score(prefix, include_last_word=False)

    def _final_score(self, prefix, probs):
        score = _logadd(*probs) + self.token_bonus * len(prefix)
        return score + self._lm_prefix_score(prefix, include_last_word=True)

    def _lm_prefix_score(self, prefix, include_last_word):
        if self.lm is None or (self.lm_weight == 0.0 and self.word_bonus == 0.0):
            return 0.0

        key = (prefix, include_last_word)
        if key in self._lm_score_cache:
            return self._lm_score_cache[key]

        words = self._prefix_words(prefix, include_last_word)
        if not words:
            score = 0.0
        else:
            score = self.lm_weight * self.lm.score_words(words, add_eos=include_last_word)
            score += self.word_bonus * len(words)
        self._lm_score_cache[key] = score
        return score

    def _prefix_words(self, prefix, include_last_word):
        if not prefix:
            return []

        if include_last_word:
            return self.lm.words(self.tokenizer.decode(list(prefix)))

        pieces = [self.tokenizer.id_to_piece(int(token)) for token in prefix]
        word_starts = [idx for idx, piece in enumerate(pieces) if piece.startswith("▁")]
        if len(word_starts) <= 1:
            return []

        current_word_start = word_starts[-1]
        completed_prefix = prefix[:current_word_start]
        return self.lm.words(self.tokenizer.decode(list(completed_prefix)))


class FlashlightUnigramLM:
    def __init__(self, word_lm, lm_dict_words):
        from torchaudio.models.decoder import CTCDecoderLM, CTCDecoderLMState

        class _LM(CTCDecoderLM):
            def __init__(self, outer):
                super().__init__()
                self.outer = outer

            def start(self, start_with_nothing):
                del start_with_nothing
                return CTCDecoderLMState()

            def score(self, state, usr_token_idx):
                word = self.outer.word_for_index(usr_token_idx)
                # Flashlight's Python binding returns internal LMState objects
                # during decoding, so Python-side histories are not reliable here.
                # Use a robust unigram LM score while still applying it during
                # beam expansion.
                score = self.outer.word_lm.score_next((), word)
                return state.child(int(usr_token_idx)), score

            def finish(self, state):
                score = self.outer.word_lm.score_next((), "</s>")
                return state, score

        self.word_lm = word_lm
        self.lm_dict_words = list(lm_dict_words)
        self.decoder_lm = _LM(self)

    def word_for_index(self, usr_token_idx):
        if 0 <= int(usr_token_idx) < len(self.lm_dict_words):
            word = self.lm_dict_words[int(usr_token_idx)]
            return word if word in self.word_lm.vocab else "<unk>"
        return "<unk>"


class FlashlightCTCDecoder(nn.Module):
    """Flashlight beam search with either a Python LM or an external KenLM."""

    def __init__(
        self,
        tokenizer_path,
        lm,
        beam_size=50,
        cutoff_top_n=40,
        temperature=1.0,
        lm_weight=0.6,
        word_bonus=0.0,
        asset_dir=None,
        kenlm_path=None,
        unigram_path=None,
        max_words=200000,
    ):
        super().__init__()
        from torchaudio.models.decoder import ctc_decoder

        self.tokenizer = spm.SentencePieceProcessor(tokenizer_path)
        self.temperature = float(temperature)
        self.labels = ["-"] + [self.tokenizer.id_to_piece(idx) for idx in range(1, self.tokenizer.vocab_size())]
        self.word_lm = lm
        self.kenlm_path = kenlm_path
        self.unigram_path = unigram_path
        self.max_words = int(max_words)
        self.asset_dir = asset_dir or os.path.join(".tmp", "voxmm_flashlight_decoder")
        os.makedirs(self.asset_dir, exist_ok=True)
        self.lm_words = self._decoder_words(lm, unigram_path, self.max_words)
        lexicon_path, lm_dict_path = self._write_assets()
        flashlight_lm = kenlm_path if kenlm_path else FlashlightUnigramLM(lm, self.lm_words).decoder_lm

        self.decoder = ctc_decoder(
            lexicon=lexicon_path,
            tokens=self.labels,
            lm=flashlight_lm,
            lm_dict=lm_dict_path,
            nbest=1,
            beam_size=int(beam_size),
            beam_size_token=int(cutoff_top_n),
            lm_weight=float(lm_weight),
            word_score=float(word_bonus),
            unk_score=NEG_INF,
            sil_score=0.0,
            log_add=False,
            blank_token="-",
            sil_token="-",
            unk_word="<unk>",
        )

    def forward(self, outputs, from_logits=True):
        if from_logits:
            tokens = self.beam_search(*outputs)
        else:
            tokens = outputs[0].tolist()
        return self.tokenizer.decode(tokens)

    def beam_search(self, logits, logits_len):
        log_probs = (logits.detach().float().cpu() / self.temperature).log_softmax(dim=-1)
        lengths = logits_len.detach().cpu().to(torch.int32)
        hypotheses = self.decoder(log_probs, lengths)
        return [self._hypothesis_tokens(item[0]) if item else [] for item in hypotheses]

    def _hypothesis_tokens(self, hypothesis):
        tokens = hypothesis.tokens
        if hasattr(tokens, "tolist"):
            tokens = tokens.tolist()
        return [int(token) for token in tokens if int(token) != 0]

    def _decoder_words(self, lm, unigram_path, max_words):
        if unigram_path:
            if not os.path.exists(unigram_path):
                raise FileNotFoundError(
                    f"VOXMM_UNIGRAM_PATH points to {unigram_path!r}, but that file does not exist."
                )
            words_by_spelling = {}
            truncated = False

            def usable_word(line):
                word = line.strip().split()[0] if line.strip() else ""
                if not word or word.startswith("#") or word.lower() in {"<s>", "</s>", "<unk>"}:
                    return None
                return word

            with open(unigram_path, encoding="utf-8") as handle:
                for line in handle:
                    word = usable_word(line)
                    if word is None:
                        continue
                    words_by_spelling.setdefault(word.lower(), word)
                    if max_words > 0 and len(words_by_spelling) >= max_words:
                        truncated = any(usable_word(rest) is not None for rest in handle)
                        break
            if truncated:
                warnings.warn(
                    f"VOXMM_KENLM_MAX_WORDS={max_words} truncated {unigram_path!r}. "
                    "Words outside the truncated lexicon cannot be emitted by Flashlight; "
                    "use 0 or the full vocabulary size for reliable KenLM evaluation.",
                    RuntimeWarning,
                )
            words = [words_by_spelling[spelling] for spelling in sorted(words_by_spelling)]
        else:
            words = sorted(word for word in lm.vocab if word not in {"<unk>", "</s>"})
        return ["<unk>"] + words

    def _write_assets(self):
        lexicon_path = os.path.join(self.asset_dir, "lexicon.txt")
        lm_dict_path = os.path.join(self.asset_dir, "lm_dict.txt")

        with open(lm_dict_path, "w", encoding="utf-8") as handle:
            for word in self.lm_words:
                handle.write(f"{word}\n")

        with open(lexicon_path, "w", encoding="utf-8") as handle:
            for word in self.lm_words:
                if word == "<unk>":
                    continue
                token_ids = self.tokenizer.encode(word.lower())
                if not token_ids or any(token_id == 0 for token_id in token_ids):
                    continue
                pieces = [self.tokenizer.id_to_piece(int(token_id)) for token_id in token_ids]
                handle.write(f"{word}\t{' '.join(pieces)}\n")

        return lexicon_path, lm_dict_path


def _make_lm(options):
    lm_manifest = options.get("lm_manifest")
    if not lm_manifest:
        return None
    if not os.path.exists(lm_manifest):
        raise FileNotFoundError(
            f"VOXMM_LM_MANIFEST points to {lm_manifest!r}, but that file does not exist."
        )

    lm_type = options.get("lm_type", "kneser_ney")
    if lm_type in {"kneser_ney", "kn", "modified_kneser_ney"}:
        return KneserNeyWordNGramLM(
            manifest_path=lm_manifest,
            order=options["lm_order"],
            discount=options["lm_discount"],
        )
    if lm_type in {"add_k", "simple", "laplace"}:
        return SimpleWordNGramLM(
            manifest_path=lm_manifest,
            order=options["lm_order"],
            smoothing=options["lm_smoothing"],
        )
    raise ValueError(f"Unsupported VOXMM_LM_TYPE={lm_type!r}")


def decoder_options_from_env(prefix="VOXMM"):
    return {
        "decoder": os.environ.get(f"{prefix}_DECODER", "greedy").lower(),
        "beam_size": int(os.environ.get(f"{prefix}_BEAM_SIZE", "16")),
        "beam_cutoff_top_n": int(os.environ.get(f"{prefix}_BEAM_CUTOFF_TOP_N", "40")),
        "beam_temperature": float(os.environ.get(f"{prefix}_BEAM_TEMPERATURE", "1.0")),
        "beam_token_bonus": float(os.environ.get(f"{prefix}_BEAM_TOKEN_BONUS", "0.0")),
        "lm_manifest": os.environ.get(f"{prefix}_LM_MANIFEST") or os.environ.get(f"{prefix}_TRAIN_MANIFEST"),
        "lm_weight": float(os.environ.get(f"{prefix}_LM_WEIGHT", "0.0")),
        "lm_type": os.environ.get(f"{prefix}_LM_TYPE", "kneser_ney").lower(),
        "lm_order": int(os.environ.get(f"{prefix}_LM_ORDER", "4")),
        "lm_smoothing": float(os.environ.get(f"{prefix}_LM_SMOOTHING", "0.1")),
        "lm_discount": float(os.environ.get(f"{prefix}_LM_DISCOUNT", "0.75")),
        "word_bonus": float(os.environ.get(f"{prefix}_WORD_BONUS", "0.0")),
        "flashlight_asset_dir": os.environ.get(f"{prefix}_FLASHLIGHT_ASSET_DIR", os.path.join(".tmp", "voxmm_flashlight_decoder")),
        "kenlm_path": os.environ.get(f"{prefix}_KENLM_PATH") or os.environ.get(f"{prefix}_NGRAM_PATH"),
        "unigram_path": os.environ.get(f"{prefix}_UNIGRAM_PATH"),
        "kenlm_max_words": int(os.environ.get(f"{prefix}_KENLM_MAX_WORDS", "200000")),
        "beam_num_processes": int(os.environ.get(f"{prefix}_BEAM_NUM_PROCESSES", "8")),
        "ngram_path": os.environ.get(f"{prefix}_NGRAM_PATH"),
        "ngram_alpha": float(os.environ.get(f"{prefix}_NGRAM_ALPHA", "0.6")),
        "ngram_beta": float(os.environ.get(f"{prefix}_NGRAM_BETA", "1.0")),
        "ngram_tmp": float(os.environ.get(f"{prefix}_NGRAM_TMP", "1.0")),
        "ngram_offset": int(os.environ.get(f"{prefix}_NGRAM_OFFSET", "100")),
        "neural_config_path": os.environ.get(f"{prefix}_NEURAL_CONFIG_PATH"),
        "neural_checkpoint": os.environ.get(f"{prefix}_NEURAL_CHECKPOINT"),
        "neural_alpha": float(os.environ.get(f"{prefix}_NEURAL_ALPHA", "0.6")),
        "neural_beta": float(os.environ.get(f"{prefix}_NEURAL_BETA", "1.0")),
    }


def make_ctc_decoder(nnet, tokenizer_path, options=None):
    options = decoder_options_from_env() if options is None else options
    decoder = options["decoder"]

    if decoder in {"greedy", "ctc_greedy"}:
        return nnet.CTCGreedySearchDecoder(tokenizer_path=tokenizer_path)

    if decoder in {"prefix_beam", "beam", "beam_search", "prefix_beam_lm", "beam_lm"}:
        lm = _make_lm(options) if decoder in {"prefix_beam_lm", "beam_lm"} or options["lm_weight"] != 0.0 else None
        return CTCPrefixBeamSearchDecoder(
            tokenizer_path=tokenizer_path,
            beam_size=options["beam_size"],
            cutoff_top_n=options["beam_cutoff_top_n"],
            temperature=options["beam_temperature"],
            token_bonus=options["beam_token_bonus"],
            lm=lm,
            lm_weight=options["lm_weight"],
            word_bonus=options["word_bonus"],
        )

    if decoder in {"flashlight_lm", "torchaudio_lm"}:
        lm = _make_lm(options)
        return FlashlightCTCDecoder(
            tokenizer_path=tokenizer_path,
            lm=lm,
            beam_size=options["beam_size"],
            cutoff_top_n=options["beam_cutoff_top_n"],
            temperature=options["beam_temperature"],
            lm_weight=options["lm_weight"],
            word_bonus=options["word_bonus"],
            asset_dir=options["flashlight_asset_dir"],
        )

    if decoder in {"flashlight_kenlm", "kenlm", "hf_kenlm"}:
        kenlm_path = options.get("kenlm_path")
        if not kenlm_path:
            raise ValueError("VOXMM_DECODER=flashlight_kenlm requires VOXMM_KENLM_PATH or VOXMM_NGRAM_PATH.")
        if not os.path.exists(kenlm_path):
            raise FileNotFoundError(f"KenLM path {kenlm_path!r} does not exist.")
        lm = _make_lm(options)
        return FlashlightCTCDecoder(
            tokenizer_path=tokenizer_path,
            lm=lm,
            beam_size=options["beam_size"],
            cutoff_top_n=options["beam_cutoff_top_n"],
            temperature=options["beam_temperature"],
            lm_weight=options["lm_weight"],
            word_bonus=options["word_bonus"],
            asset_dir=options["flashlight_asset_dir"],
            kenlm_path=kenlm_path,
            unigram_path=options.get("unigram_path"),
            max_words=options["kenlm_max_words"],
        )

    if decoder in {"ctcdecode", "ngram_beam"}:
        try:
            import ctcdecode  # noqa: F401
        except Exception as exc:
            raise RuntimeError(
                "VOXMM_DECODER=ctcdecode requires the 'ctcdecode' Python package. "
                "Use VOXMM_DECODER=prefix_beam_lm for dependency-free LM rescoring, "
                "or install ctcdecode and provide VOXMM_NGRAM_PATH for upstream LM decoding."
            ) from exc

        return nnet.CTCBeamSearchDecoder(
            tokenizer_path=tokenizer_path,
            beam_size=options["beam_size"],
            ngram_path=options["ngram_path"],
            ngram_tmp=options["ngram_tmp"],
            ngram_alpha=options["ngram_alpha"],
            ngram_beta=options["ngram_beta"],
            ngram_offset=options["ngram_offset"],
            neural_config_path=options["neural_config_path"],
            neural_checkpoint=options["neural_checkpoint"],
            neural_alpha=options["neural_alpha"],
            neural_beta=options["neural_beta"],
            num_processes=options["beam_num_processes"],
        )

    raise ValueError(f"Unsupported VOXMM_DECODER={decoder!r}")
