# Copyright FunASR (https://github.com/modelscope/FunASR). All Rights Reserved.
#  MIT License  (https://opensource.org/licenses/MIT)

"""Text-level hotword correction after ASR decoding.

This module is intentionally separate from model-level ``hotword`` / ``hotwords``
prompting. It runs after ASR (and punctuation / ITN when configured) and only
updates top-level ``text`` plus sentence-level ``text`` / ``sentence`` fields.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

HotwordInput = Union[str, Sequence[str], Mapping[str, str], None]

_EXPLICIT_SEPARATORS = ("=>", "->", "→")
_TOKEN_PATTERN = re.compile(r"[\u4e00-\u9fff]|[a-zA-Z]+|[0-9]+")

_LAZY_PINYIN = None
_PINYIN_STYLE = None
_RAPIDFUZZ_FUZZ = None


@dataclass(frozen=True)
class HotwordMatch:
    """A single postprocess hotword replacement."""

    original: str
    replacement: str
    score: float
    start: int
    end: int

    def as_dict(self) -> Dict[str, Any]:
        return {
            "original": self.original,
            "replacement": self.replacement,
            "score": self.score,
            "start": self.start,
            "end": self.end,
        }


def _require_pypinyin():
    global _LAZY_PINYIN, _PINYIN_STYLE
    if _LAZY_PINYIN is None:
        try:
            from pypinyin import Style, lazy_pinyin
        except ImportError as exc:
            raise ImportError(
                "postprocess hotword fuzzy matching requires pypinyin. "
                "Install it with: pip install pypinyin"
            ) from exc
        _LAZY_PINYIN = lazy_pinyin
        _PINYIN_STYLE = Style
    return _LAZY_PINYIN, _PINYIN_STYLE


def _require_rapidfuzz():
    global _RAPIDFUZZ_FUZZ
    if _RAPIDFUZZ_FUZZ is None:
        try:
            from rapidfuzz import fuzz
        except ImportError as exc:
            raise ImportError(
                "postprocess hotword fuzzy matching requires rapidfuzz. "
                "Install it with: pip install rapidfuzz"
            ) from exc
        _RAPIDFUZZ_FUZZ = fuzz
    return _RAPIDFUZZ_FUZZ


def _to_pinyin_key(text: str) -> str:
    lazy_pinyin, style = _require_pypinyin()
    return "".join(lazy_pinyin(text, style=style.NORMAL, errors="ignore")).lower()


def _parse_line(line: str) -> Tuple[Optional[str], Optional[str], bool]:
    """Parse one hotword file line.

    Returns:
        (wrong, right, is_explicit)
        For fuzzy-only targets, wrong is None and right is the target word.
    """
    stripped = line.strip()
    if not stripped or stripped.startswith("#"):
        return None, None, False

    for sep in _EXPLICIT_SEPARATORS:
        if sep in stripped:
            wrong, right = stripped.split(sep, 1)
            wrong = wrong.strip()
            right = right.strip()
            if wrong and right:
                return wrong, right, True
            return None, None, False

    return None, stripped, False


def parse_hotword_file(path: str) -> Tuple[Dict[str, str], List[str]]:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"postprocess_hotword_file not found: {path}")

    explicit: Dict[str, str] = {}
    fuzzy_targets: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            wrong, right, is_explicit = _parse_line(line)
            if not right:
                continue
            if is_explicit and wrong is not None:
                explicit[wrong] = right
            else:
                fuzzy_targets.append(right)
    return explicit, fuzzy_targets


def parse_postprocess_hotwords(
    postprocess_hotwords: HotwordInput,
) -> Tuple[Dict[str, str], List[str]]:
    """Parse in-memory hotword config into explicit and fuzzy buckets."""
    explicit: Dict[str, str] = {}
    fuzzy_targets: List[str] = []

    if postprocess_hotwords is None:
        return explicit, fuzzy_targets

    if isinstance(postprocess_hotwords, str):
        for line in postprocess_hotwords.splitlines():
            wrong, right, is_explicit = _parse_line(line)
            if not right:
                continue
            if is_explicit and wrong is not None:
                explicit[wrong] = right
            else:
                fuzzy_targets.append(right)
        return explicit, fuzzy_targets

    if isinstance(postprocess_hotwords, Mapping):
        for wrong, right in postprocess_hotwords.items():
            wrong_s = str(wrong).strip()
            right_s = str(right).strip()
            if not right_s:
                continue
            if wrong_s and wrong_s != right_s:
                explicit[wrong_s] = right_s
            else:
                fuzzy_targets.append(right_s)
        return explicit, fuzzy_targets

    if isinstance(postprocess_hotwords, Sequence) and not isinstance(postprocess_hotwords, (str, bytes)):
        for item in postprocess_hotwords:
            if item is None:
                continue
            item_s = str(item).strip()
            if not item_s:
                continue
            wrong, right, is_explicit = _parse_line(item_s)
            if is_explicit and wrong is not None:
                explicit[wrong] = right
            elif right:
                fuzzy_targets.append(right)
        return explicit, fuzzy_targets

    raise TypeError(
        "postprocess_hotwords must be None, str, list, or dict; "
        f"got {type(postprocess_hotwords)!r}"
    )


def build_postprocess_hotword_matcher(
    postprocess_hotwords: HotwordInput = None,
    postprocess_hotword_file: Optional[str] = None,
    postprocess_hotword_threshold: float = 0.85,
    enable_fuzzy: bool = True,
) -> Optional["PostprocessHotwordMatcher"]:
    """Compile a matcher once per ``generate()`` call."""
    explicit: Dict[str, str] = {}
    fuzzy_targets: List[str] = []

    if postprocess_hotwords is not None:
        e, f = parse_postprocess_hotwords(postprocess_hotwords)
        explicit.update(e)
        fuzzy_targets.extend(f)

    if postprocess_hotword_file:
        e, f = parse_hotword_file(postprocess_hotword_file)
        explicit.update(e)
        fuzzy_targets.extend(f)

    if not explicit and not fuzzy_targets:
        return None

    return PostprocessHotwordMatcher(
        explicit_map=explicit,
        fuzzy_targets=fuzzy_targets,
        threshold=postprocess_hotword_threshold,
        enable_fuzzy=enable_fuzzy,
    )


class PostprocessHotwordMatcher:
    """Compiled matcher reused across all results in one generate() call."""

    def __init__(
        self,
        explicit_map: Optional[Dict[str, str]] = None,
        fuzzy_targets: Optional[Iterable[str]] = None,
        threshold: float = 0.85,
        enable_fuzzy: bool = True,
    ):
        self.explicit_map = dict(explicit_map or {})
        self.threshold = float(threshold)
        if not 0.0 <= self.threshold <= 1.0:
            raise ValueError(
                f"postprocess_hotword_threshold must be between 0.0 and 1.0, got {threshold}"
            )
        self.enable_fuzzy = bool(enable_fuzzy)

        seen = set()
        self.fuzzy_targets: List[str] = []
        for target in fuzzy_targets or []:
            target_s = str(target).strip()
            if target_s and target_s not in seen:
                seen.add(target_s)
                self.fuzzy_targets.append(target_s)

        self._length_buckets: Dict[int, List[Tuple[str, str]]] = {}
        self._fuzz = None
        if self.fuzzy_targets and self.enable_fuzzy:
            self._fuzz = _require_rapidfuzz()
            _require_pypinyin()
            for target in self.fuzzy_targets:
                bucket = self._length_buckets.setdefault(len(target), [])
                bucket.append((target, _to_pinyin_key(target)))

    def apply_text(self, text: str) -> Tuple[str, List[HotwordMatch]]:
        if not text:
            return text, []

        matches: List[HotwordMatch] = []
        updated = self._apply_explicit(text, matches)
        if self.fuzzy_targets and self.enable_fuzzy:
            updated, fuzzy_matches = self._apply_fuzzy(updated)
            matches.extend(fuzzy_matches)
        return updated, matches

    def apply_result(self, result: Dict[str, Any], return_matches: bool = False) -> Dict[str, Any]:
        text = result.get("text", "")
        if not isinstance(text, str) or not text:
            if return_matches:
                result["postprocess_hotword_matches"] = []
            return result

        original_timestamp = result.get("timestamp")
        new_text, matches = self.apply_text(text)
        result["text"] = new_text

        sentence_info = result.get("sentence_info")
        if isinstance(sentence_info, list):
            for sentence in sentence_info:
                if not isinstance(sentence, dict):
                    continue
                for field in ("text", "sentence"):
                    if field in sentence and isinstance(sentence[field], str):
                        corrected, _ = self.apply_text(sentence[field])
                        sentence[field] = corrected

        if return_matches:
            result["postprocess_hotword_matches"] = [m.as_dict() for m in matches]

        # Timestamps intentionally remain aligned to the original recognition.
        if original_timestamp is not None:
            result["timestamp"] = original_timestamp
        return result

    def _apply_explicit(self, text: str, matches: List[HotwordMatch]) -> str:
        if not self.explicit_map:
            return text

        updated = text
        for wrong in sorted(self.explicit_map, key=len, reverse=True):
            right = self.explicit_map[wrong]
            start = 0
            while True:
                idx = updated.find(wrong, start)
                if idx < 0:
                    break
                end = idx + len(wrong)
                matches.append(
                    HotwordMatch(
                        original=wrong,
                        replacement=right,
                        score=1.0,
                        start=idx,
                        end=end,
                    )
                )
                updated = updated[:idx] + right + updated[end:]
                start = idx + len(right)
        return updated

    def _apply_fuzzy(self, text: str) -> Tuple[str, List[HotwordMatch]]:
        assert self._fuzz is not None
        candidates: List[HotwordMatch] = []

        if not self._length_buckets:
            return text, []

        min_len = min(self._length_buckets)
        max_len = max(self._length_buckets)
        text_len = len(text)

        for win_len in range(max(1, min_len - 1), max_len + 2):
            bucket_keys = [
                length
                for length in (win_len - 1, win_len, win_len + 1)
                if length in self._length_buckets
            ]
            if not bucket_keys:
                continue

            for start in range(0, text_len - win_len + 1):
                end = start + win_len
                segment = text[start:end]
                if not segment or not _TOKEN_PATTERN.search(segment):
                    continue
                segment_py = _to_pinyin_key(segment)

                for length in bucket_keys:
                    for target, target_py in self._length_buckets[length]:
                        if segment == target:
                            continue
                        score = self._fuzz.ratio(segment_py, target_py) / 100.0
                        if score >= self.threshold:
                            candidates.append(
                                HotwordMatch(
                                    original=segment,
                                    replacement=target,
                                    score=round(score, 4),
                                    start=start,
                                    end=end,
                                )
                            )

        if not candidates:
            return text, []

        selected = _select_non_overlapping(candidates)
        updated = text
        applied: List[HotwordMatch] = []
        for match in sorted(selected, key=lambda m: m.start, reverse=True):
            updated = updated[: match.start] + match.replacement + updated[match.end :]
            applied.append(match)
        applied.sort(key=lambda m: m.start)
        return updated, applied


def _select_non_overlapping(candidates: List[HotwordMatch]) -> List[HotwordMatch]:
    ranked = sorted(candidates, key=lambda m: (m.score, m.end - m.start), reverse=True)
    selected: List[HotwordMatch] = []
    occupied: List[Tuple[int, int]] = []

    for candidate in ranked:
        if any(not (candidate.end <= start or candidate.start >= end) for start, end in occupied):
            continue
        selected.append(candidate)
        occupied.append((candidate.start, candidate.end))

    return sorted(selected, key=lambda m: m.start)


def apply_postprocess_hotwords_to_results(
    results: List[Dict[str, Any]],
    cfg: Mapping[str, Any],
) -> List[Dict[str, Any]]:
    """Apply compiled matcher to each result dict if configured in cfg."""
    matcher = build_postprocess_hotword_matcher(
        postprocess_hotwords=cfg.get("postprocess_hotwords"),
        postprocess_hotword_file=cfg.get("postprocess_hotword_file"),
        postprocess_hotword_threshold=cfg.get("postprocess_hotword_threshold", 0.85),
        enable_fuzzy=cfg.get("postprocess_hotword_fuzzy", True),
    )
    if matcher is None:
        return results

    return_matches = bool(
        cfg.get("return_postprocess_hotword_matches", False)
    )
    for result in results:
        if isinstance(result, dict):
            matcher.apply_result(result, return_matches=return_matches)
    return results
