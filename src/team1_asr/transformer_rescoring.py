from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


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


@dataclass(frozen=True)
class RerankSettings:
    top_k: int = 50
    transformer_weight: float = 0.16
    word_bonus: float = 1.30
    base_weight: float = 1.0
    drop_penalty: float = -1.0


@dataclass(frozen=True)
class RerankResult:
    text: str
    first_pass_text: str
    selected_rank: int
    first_pass_score: float
    selected_score: float
    candidates: list[dict[str, object]]

    def metadata(self, *, candidate_limit: int = 5) -> dict[str, object]:
        return {
            "first_pass_transcript": self.first_pass_text,
            "selected_rank": self.selected_rank,
            "first_pass_score": self.first_pass_score,
            "selected_score": self.selected_score,
            "top_candidates": self.candidates[: max(0, candidate_limit)],
        }


def normalize_hypothesis(
    text: str,
    dropped_words: Iterable[str] = FILLED_PAUSE_WORDS,
) -> tuple[str, int]:
    dropped = {word.lower() for word in dropped_words}
    words = text.split()
    kept = [word for word in words if word.lower() not in dropped]
    return " ".join(kept), len(words) - len(kept)


def rerank_scored_candidates(
    candidates: list[dict[str, object]],
    settings: RerankSettings,
    *,
    dropped_words: Iterable[str] = FILLED_PAUSE_WORDS,
) -> RerankResult:
    if not candidates:
        raise ValueError("Cannot rerank an empty CTC candidate list.")
    limited = candidates[: max(1, settings.top_k)]
    annotated: list[dict[str, object]] = []
    for rank, candidate in enumerate(limited, start=1):
        text = str(candidate.get("text") or "").strip()
        normalized, dropped_count = normalize_hypothesis(text, dropped_words)
        acoustic_lm_score = float(candidate.get("score") or 0.0)
        transformer_score = float(candidate.get("transformer_lm_score") or 0.0)
        combined = (
            settings.base_weight * acoustic_lm_score
            + settings.transformer_weight * transformer_score
            + settings.word_bonus * len(normalized.split())
            - settings.drop_penalty * dropped_count
        )
        annotated.append(
            {
                **candidate,
                "rank": rank,
                "normalized_text": normalized,
                "dropped_word_count": dropped_count,
                "combined_score": combined,
            }
        )

    selected = max(annotated, key=lambda item: float(item["combined_score"]))
    first = annotated[0]
    return RerankResult(
        text=str(selected["normalized_text"]),
        first_pass_text=str(first.get("text") or "").strip(),
        selected_rank=int(selected["rank"]),
        first_pass_score=float(first.get("score") or 0.0),
        selected_score=float(selected["combined_score"]),
        candidates=annotated,
    )


class RobertaPseudoLikelihoodRescorer:
    """Persistent masked-LM scorer for online CTC N-best reranking."""

    def __init__(
        self,
        *,
        model_name: str = "FacebookAI/roberta-base",
        device: str = "auto",
        batch_size: int = 32,
        cache_dir: str | Path | None = None,
    ) -> None:
        import torch
        import torch.nn.functional as functional
        from transformers import AutoModelForMaskedLM, AutoTokenizer

        self._torch = torch
        self._functional = functional
        self.model_name = model_name
        self.batch_size = max(1, int(batch_size))
        selected_device = (
            "cuda" if device == "auto" and torch.cuda.is_available() else device
        )
        if selected_device == "auto":
            selected_device = "cpu"
        self.device = torch.device(selected_device)
        pretrained_kwargs = {}
        if cache_dir is not None:
            pretrained_kwargs["cache_dir"] = str(Path(cache_dir))
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            **pretrained_kwargs,
        )
        self.model = AutoModelForMaskedLM.from_pretrained(
            model_name,
            **pretrained_kwargs,
        ).to(self.device)
        self.model.eval()
        if self.tokenizer.pad_token_id is None:
            raise ValueError(f"Masked LM {model_name!r} has no pad token.")
        if self.tokenizer.mask_token_id is None:
            raise ValueError(f"Masked LM {model_name!r} has no mask token.")

    def rerank(
        self,
        candidates: list[dict[str, object]],
        settings: RerankSettings,
        *,
        dropped_words: Iterable[str] = FILLED_PAUSE_WORDS,
    ) -> RerankResult:
        limited = candidates[: max(1, settings.top_k)]
        normalized = [
            normalize_hypothesis(str(candidate.get("text") or ""), dropped_words)[0]
            for candidate in limited
        ]
        scores = self.score_texts(normalized)
        annotated = [
            {**candidate, "transformer_lm_score": scores[text]}
            for candidate, text in zip(limited, normalized)
        ]
        return rerank_scored_candidates(
            annotated,
            settings,
            dropped_words=dropped_words,
        )

    def score_texts(self, texts: Iterable[str]) -> dict[str, float]:
        unique_texts = list(dict.fromkeys(str(text) for text in texts))
        sequences = {
            text: self.tokenizer.encode(text, add_special_tokens=True)
            for text in unique_texts
        }
        special_ids = set(self.tokenizer.all_special_ids)
        tasks = [
            (text, position, token_id)
            for text, token_ids in sequences.items()
            for position, token_id in enumerate(token_ids)
            if token_id not in special_ids
        ]
        tasks.sort(key=lambda item: len(sequences[item[0]]))
        scores = {text: 0.0 for text in unique_texts}

        with self._torch.inference_mode():
            for start in range(0, len(tasks), self.batch_size):
                batch = tasks[start : start + self.batch_size]
                self._score_batch(batch, sequences, scores)
        return scores

    def _score_batch(
        self,
        batch: list[tuple[str, int, int]],
        sequences: dict[str, list[int]],
        scores: dict[str, float],
    ) -> None:
        if not batch:
            return
        torch = self._torch
        max_length = max(len(sequences[text]) for text, _, _ in batch)
        input_ids = torch.full(
            (len(batch), max_length),
            self.tokenizer.pad_token_id,
            dtype=torch.long,
            device=self.device,
        )
        attention_mask = torch.zeros_like(input_ids)
        positions: list[int] = []
        targets: list[int] = []
        for row_index, (text, position, target) in enumerate(batch):
            token_ids = sequences[text]
            length = len(token_ids)
            input_ids[row_index, :length] = torch.tensor(
                token_ids,
                dtype=torch.long,
                device=self.device,
            )
            input_ids[row_index, position] = self.tokenizer.mask_token_id
            attention_mask[row_index, :length] = 1
            positions.append(position)
            targets.append(target)

        row_indices = torch.arange(len(batch), device=self.device)
        position_indices = torch.tensor(positions, device=self.device)
        if hasattr(self.model, "roberta") and hasattr(self.model, "lm_head"):
            hidden = self.model.roberta(
                input_ids=input_ids,
                attention_mask=attention_mask,
            ).last_hidden_state
            logits = self.model.lm_head(hidden[row_indices, position_indices])
        else:
            all_logits = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            ).logits
            logits = all_logits[row_indices, position_indices]
        target_tensor = torch.tensor(targets, device=self.device)
        token_scores = -self._functional.cross_entropy(
            logits,
            target_tensor,
            reduction="none",
        )
        for (text, _, _), value in zip(batch, token_scores.cpu().tolist()):
            scores[text] += float(value)
