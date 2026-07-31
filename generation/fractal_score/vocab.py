"""Token vocabulary for the harmonic-function refinement operator."""
from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass

from generation.fractal_score.harmony import NONE_TOKEN, HarmonyProgression

PAD_TOKEN = "<pad>"
MASK_TOKEN = "<mask>"
RARE_TOKEN = "<rare>"
SPECIAL_TOKENS = (PAD_TOKEN, MASK_TOKEN, NONE_TOKEN, RARE_TOKEN)


class VocabError(ValueError):
    """Raised when a vocabulary is misused."""


@dataclass(frozen=True)
class HarmonyVocab:
    """A frozen chord-token vocabulary with reserved control tokens."""

    tokens: tuple[str, ...]
    keys: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.tokens[: len(SPECIAL_TOKENS)] != SPECIAL_TOKENS:
            raise VocabError("vocabulary must start with the reserved control tokens")

    @classmethod
    def build(
        cls,
        progressions: Iterable[HarmonyProgression],
        *,
        max_tokens: int,
    ) -> "HarmonyVocab":
        chord_counts: Counter[str] = Counter()
        key_counts: Counter[str] = Counter()
        for progression in progressions:
            key_counts[progression.home_key] += 1
            for token in progression.tokens:
                if token == NONE_TOKEN:
                    continue
                chord_counts[token] += 1
        budget = max_tokens - len(SPECIAL_TOKENS)
        if budget < 1:
            raise VocabError("max_tokens is too small for the control tokens")
        common = [token for token, _ in chord_counts.most_common(budget)]
        tokens = (*SPECIAL_TOKENS, *sorted(common))
        keys = tuple(sorted(key_counts))
        return cls(tokens=tokens, keys=keys)

    def __len__(self) -> int:
        return len(self.tokens)

    @property
    def pad_id(self) -> int:
        return self.tokens.index(PAD_TOKEN)

    @property
    def mask_id(self) -> int:
        return self.tokens.index(MASK_TOKEN)

    @property
    def none_id(self) -> int:
        return self.tokens.index(NONE_TOKEN)

    @property
    def rare_id(self) -> int:
        return self.tokens.index(RARE_TOKEN)

    def encode(self, token: str) -> int:
        try:
            return self._token_index[token]
        except KeyError:
            return self.rare_id

    def decode(self, index: int) -> str:
        return self.tokens[index]

    def encode_key(self, key: str) -> int:
        return self._key_index.get(key, 0)

    @property
    def key_count(self) -> int:
        return len(self.keys)

    def to_dict(self) -> dict[str, list[str]]:
        return {"tokens": list(self.tokens), "keys": list(self.keys)}

    @classmethod
    def from_dict(cls, value: dict[str, list[str]]) -> "HarmonyVocab":
        return cls(tokens=tuple(value["tokens"]), keys=tuple(value["keys"]))

    @property
    def _token_index(self) -> dict[str, int]:
        cached = getattr(self, "_token_index_cache", None)
        if cached is None:
            cached = {token: index for index, token in enumerate(self.tokens)}
            object.__setattr__(self, "_token_index_cache", cached)
        return cached

    @property
    def _key_index(self) -> dict[str, int]:
        cached = getattr(self, "_key_index_cache", None)
        if cached is None:
            cached = {key: index for index, key in enumerate(self.keys)}
            object.__setattr__(self, "_key_index_cache", cached)
        return cached
