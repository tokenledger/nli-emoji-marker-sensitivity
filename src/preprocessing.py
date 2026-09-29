"""Normalization utilities for explicit inference-time experimental conditions."""

from __future__ import annotations

import re
from typing import Iterable

from transforms import (
    HELD_OUT_MARKERS,
    MAIN_EMOJI_MAP,
    TRAIN_MARKERS,
    canonicalize_whitespace,
)


SLANG_TO_STANDARD = {
    "gonna": "going to", "wanna": "want to", "tryna": "trying to",
    "kinda": "kind of", "sorta": "sort of", "outta": "out of",
    "hella": "a lot of", "gotta": "got to", "dunno": "don't know",
    "lemme": "let me", "gimme": "give me", "shoulda": "should have",
    "woulda": "would have", "coulda": "could have", "finna": "about to",
    "boutta": "about to", "hafta": "have to", "hasta": "has to",
    "fronta": "front of", "toppa": "top of", "pic": "picture",
    "mic": "microphone", "veggies": "vegetables", "comfy": "comfortable",
    "specs": "spectacles", "doggo": "dog", "kitty": "cat",
    "sammy": "sandwich", "bro": "brother", "sis": "sister",
    "hubby": "husband", "prolly": "probably", "cuz": "because",
    "w/": "with", "w/o": "without", "thru": "through", "tho": "though",
}

# These generated slang forms are also ordinary words with unrelated senses.
# A text-only normalizer cannot safely infer which sense was intended.
AMBIGUOUS_SLANG_FORMS = frozenset({"shades", "kicks", "drip", "whip", "bout"})

# Because MAIN_EMOJI_MAP is one-to-one, its canonical inverse is unambiguous.
MAIN_EMOJI_TO_TEXT = {emoji: source for source, emoji in MAIN_EMOJI_MAP.items()}
KNOWN_MARKERS = TRAIN_MARKERS + HELD_OUT_MARKERS


def _marker_pattern(markers: Iterable[str]) -> re.Pattern[str] | None:
    marker_list = tuple(markers)
    if not marker_list:
        return None
    alternatives = "|".join(
        re.escape(marker) for marker in sorted(marker_list, key=len, reverse=True)
    )
    return re.compile(rf"(?<!\w)(?:{alternatives})(?!\w)", re.IGNORECASE)


def _replace_terms(text: str, replacements: dict[str, str]) -> str:
    if not replacements:
        return text
    alternatives = "|".join(re.escape(key) for key in sorted(replacements, key=len, reverse=True))
    pattern = re.compile(rf"(?<!\w)(?:{alternatives})(?!\w)", re.IGNORECASE)
    return pattern.sub(lambda match: replacements[match.group(0).lower()], text)


def remove_markers(text: str, markers: Iterable[str] = KNOWN_MARKERS) -> str:
    """Oracle deletion of known single- or multiword markers.

    This is deliberately separate from principal stage-matched preprocessing.
    """

    pattern = _marker_pattern(markers)
    if pattern is None:
        return canonicalize_whitespace(text)
    return canonicalize_whitespace(pattern.sub(" ", text))


def remove_noise(text: str) -> str:
    """Backward-compatible alias for the known-marker deletion oracle."""

    return remove_markers(text)


def audit_marker_removal(
    noisy_text: str,
    clean_text: str | None = None,
    markers: Iterable[str] = KNOWN_MARKERS,
) -> dict:
    """Report marker-removal recall separately from exact reconstruction."""

    pattern = _marker_pattern(markers)
    occurrences = list(pattern.finditer(noisy_text)) if pattern is not None else []
    reconstructed = remove_markers(noisy_text, markers)
    remaining = list(pattern.finditer(reconstructed)) if pattern is not None else []
    removed = max(0, len(occurrences) - len(remaining))
    return {
        "marker_occurrences": len(occurrences),
        "marker_occurrences_removed": removed,
        "marker_removal_recall": removed / len(occurrences) if occurrences else 1.0,
        "exact_reconstruction": (
            canonicalize_whitespace(reconstructed) == canonicalize_whitespace(clean_text)
            if clean_text is not None else None
        ),
        "reconstructed_text": reconstructed,
    }


def normalize_slang(text: str) -> str:
    return canonicalize_whitespace(_replace_terms(text, SLANG_TO_STANDARD))


def normalize_emoji(text: str) -> str:
    transformed = text
    for emoji, source in sorted(MAIN_EMOJI_TO_TEXT.items(), key=lambda item: len(item[0]), reverse=True):
        transformed = transformed.replace(emoji, source)
    return canonicalize_whitespace(transformed)


def preprocess_text(
    text: str,
    deslang: bool = True,
    deemoji: bool = True,
    denoise: bool = True,
) -> str:
    """Legacy configurable normalizer.

    Revised principal systems should call ``normalize_emoji`` directly. Setting
    ``denoise`` invokes the explicitly named known-marker deletion oracle.
    """

    if denoise:
        text = remove_markers(text)
    if deslang:
        text = normalize_slang(text)
    if deemoji:
        text = normalize_emoji(text)
    return canonicalize_whitespace(text)


def preprocess_example(example, mode: str = "all") -> dict:
    result = dict(example)
    if mode == "slang":
        transform = normalize_slang
    elif mode == "emoji":
        transform = normalize_emoji
    elif mode in {"noise", "marker_oracle"}:
        transform = remove_markers
    elif mode in {"all", "both"}:
        transform = lambda text: preprocess_text(text, True, True, True)
    elif mode == "stage_matched":
        transform = normalize_emoji
    else:
        raise ValueError(f"Unknown preprocessing mode: {mode}")

    result["premise"] = transform(example["premise"])
    result["hypothesis"] = transform(example["hypothesis"])
    return result
