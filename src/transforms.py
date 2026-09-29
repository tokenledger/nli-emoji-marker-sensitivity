"""Deterministic text transformations for the revised informal-NLI study.

Principal transformations are deliberately separated from lossy stress tests and
from marker controls. Public helpers retain the old names used by existing scripts,
but new callers can request structured provenance and supply a local RNG/seed.
"""

from __future__ import annotations

import random
import re
from dataclasses import asdict, dataclass
from typing import Mapping, MutableMapping, Sequence


# Legacy slang remains available for appendix/backward-compatible conditions. It
# is not part of the revised paper's principal emoji/marker causal design.
PHRASE_SLANG = {
    "going to": "gonna", "want to": "wanna", "trying to": "tryna",
    "kind of": "kinda", "sort of": "sorta", "out of": "outta",
    "a lot of": "hella", "got to": "gotta", "don't know": "dunno",
    "let me": "lemme", "give me": "gimme", "should have": "shoulda",
    "would have": "woulda", "could have": "coulda", "is about to": "finna",
    "getting ready to": "finna", "about to": "boutta", "have to": "hafta",
    "has to": "hasta", "in front of": "in fronta", "on top of": "on toppa",
    "next to": "nexta", "because of": "cuz of", "looking at": "lookin at",
    "talking to": "talkin to", "walking to": "walkin to",
    "do not": "dont", "does not": "doesnt", "is not": "isnt",
    "are not": "arent", "cannot": "cant", "will not": "wont",
    "did not": "didnt", "a little bit": "lil bit", "a little": "lil",
    "hold on": "hol up",
}

WORD_SLANG = {
    "picture": "pic", "microphone": "mic", "vegetables": "veggies",
    "comfortable": "comfy", "sunglasses": "shades", "spectacles": "specs",
    "shoes": "kicks", "clothes": "drip", "car": "whip", "dog": "doggo",
    "cat": "kitty", "sandwich": "sammy", "brother": "bro", "sister": "sis",
    "husband": "hubby", "probably": "prolly", "because": "cuz",
    "with": "w/", "without": "w/o", "through": "thru", "though": "tho",
    "about": "bout",
}


# Principal emoji replacements are concrete, singular nouns with unique outputs.
# Plurals are intentionally not transformed, so grammatical number is preserved.
MAIN_EMOJI_MAP = {
    "person": "👤",
    "man": "👨",
    "woman": "👩",
    "boy": "👦",
    "girl": "👧",
    "child": "🧒",
    "baby": "👶",
    "dog": "🐕",
    "cat": "🐈",
    "horse": "🐎",
    "bird": "🐦",
    "car": "🚗",
    "bicycle": "🚲",
    "camera": "📷",
    "guitar": "🎸",
    "hat": "🎩",
}


# The original mappings are preserved only as an explicitly lossy appendix stress
# test. They must never be selected by the default principal transformation.
LOSSY_EMOJI_MAP = {
    "people": "👥", "men": "👨", "women": "👩", "boys": "👦",
    "girls": "👧", "children": "👶", "kid": "🧒", "kids": "🧒",
    "guy": "👨", "guys": "👨", "dogs": "🐶", "cats": "🐱",
    "running": "🏃", "walking": "🚶", "jumping": "🤾", "playing": "⚽",
    "sitting": "🪑", "standing": "🧍", "eating": "🍽️", "drinking": "🥤",
    "sleeping": "😴", "riding": "🚴", "wearing": "👕", "holding": "✋",
    "looking": "👀", "watching": "👀", "happy": "😊", "smiling": "🙂",
    "laughing": "😂", "sad": "😢", "crying": "😭", "beach": "🏖️",
    "ocean": "🌊", "water": "💧", "snow": "❄️", "mountain": "⛰️",
    "street": "🛣️", "park": "🏞️", "field": "🌾", "ball": "⚽",
    "bike": "🚲", "food": "🍔", "shirt": "👕",
}


# Six markers are available to augmentation; three remain completely held out for
# evaluation. Human validation in Phase 3 determines which candidates survive in
# the final main set. Potentially proposition-changing expressions remain excluded.
TRAIN_MARKERS = ("fr", "tbh", "ngl", "real talk", "on god", "istg")
HELD_OUT_MARKERS = ("frfr", "deadass", "no cap")
EXCLUDED_MARKERS = ("lowkey", "highkey", "bet", "tho", "rn", "ong")
# Both control pools contain one- and two-token choices so a generator can match
# the whitespace-token length of the informal marker selected for a row.
FORMAL_MARKER_CONTROLS = ("honestly", "seriously", "in fact")
RANDOM_PHRASE_CONTROLS = ("nearby", "outside", "in the", "with it", "at this")
PSYCH_MARKER = "psych"
PSYCH_LABEL_MAP = {0: 2, 1: 1, 2: 0}
PSYCH_LABEL_POLICY = "invert_entailment_contradiction"

MARKER_POOLS = {
    "train": TRAIN_MARKERS,
    "held_out": HELD_OUT_MARKERS,
    "all": TRAIN_MARKERS + HELD_OUT_MARKERS,
    "formal": FORMAL_MARKER_CONTROLS,
    "random": RANDOM_PHRASE_CONTROLS,
}

# Compatibility aliases for older scripts. The new names above should be used by
# all revised experiments.
EMOJI_MAP = MAIN_EMOJI_MAP
NOISE_TOKENS = list(TRAIN_MARKERS + HELD_OUT_MARKERS)

MODE_TRANSFORMS = {
    "slang": frozenset(("slang",)),
    "emoji": frozenset(("emoji",)),
    "noise": frozenset(("marker",)),
    "marker": frozenset(("marker",)),
    "both": frozenset(("emoji", "marker")),
    "combined": frozenset(("emoji", "marker")),
    "lossy_emoji": frozenset(("emoji",)),
    "psych": frozenset(("psych",)),
}


@dataclass(frozen=True)
class TransformationEvent:
    """One auditable text edit, with offsets in both source and output text."""

    transform: str
    registry: str
    source: str
    replacement: str
    source_start: int
    source_end: int
    output_start: int
    output_end: int
    placement: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def canonicalize_whitespace(text: str) -> str:
    """Collapse Unicode whitespace without changing case or punctuation."""

    return " ".join(text.split())


def _resolve_rng(rng=None, seed: int | None = None):
    if rng is not None and seed is not None:
        raise ValueError("Pass either rng or seed, not both")
    if rng is not None:
        return rng
    return random.Random(seed) if seed is not None else random


def _validate_principal_registries() -> None:
    if len(MAIN_EMOJI_MAP) != len(set(MAIN_EMOJI_MAP.values())):
        raise ValueError("MAIN_EMOJI_MAP must be one-to-one")
    overlap = set(TRAIN_MARKERS) & set(HELD_OUT_MARKERS)
    if overlap:
        raise ValueError(f"Training and held-out markers overlap: {sorted(overlap)}")
    if set(EXCLUDED_MARKERS) & set(MARKER_POOLS["all"]):
        raise ValueError("Excluded markers cannot appear in a principal marker pool")


_validate_principal_registries()


def _registry_pattern(registry: Mapping[str, str]) -> re.Pattern[str] | None:
    if not registry:
        return None
    alternatives = "|".join(re.escape(key) for key in sorted(registry, key=len, reverse=True))
    return re.compile(rf"(?<!\w)(?:{alternatives})(?!\w)", re.IGNORECASE)


def _preserve_source_case(source: str, replacement: str) -> str:
    """Carry sentence-initial/all-caps casing through lexical replacements."""

    if not replacement or not any(character.isalpha() for character in replacement):
        return replacement
    if source.isupper():
        return replacement.upper()
    if source[:1].isupper():
        return replacement[:1].upper() + replacement[1:]
    return replacement


def _apply_registry(
    text: str,
    registry: Mapping[str, str],
    *,
    transform: str,
    registry_name: str,
    probability: float = 1.0,
    max_replacements: int | None = None,
    rng=None,
) -> tuple[str, list[TransformationEvent]]:
    """Apply non-overlapping lexical replacements and return exact provenance."""

    if not 0.0 <= probability <= 1.0:
        raise ValueError("probability must be between 0 and 1")
    if max_replacements is not None and max_replacements < 0:
        raise ValueError("max_replacements must be non-negative")

    pattern = _registry_pattern(registry)
    if pattern is None or max_replacements == 0:
        return text, []

    candidates = [match for match in pattern.finditer(text) if rng.random() < probability]
    if max_replacements is not None and len(candidates) > max_replacements:
        selected = set(rng.sample(range(len(candidates)), max_replacements))
        candidates = [match for index, match in enumerate(candidates) if index in selected]

    output_parts: list[str] = []
    events: list[TransformationEvent] = []
    source_cursor = 0
    output_cursor = 0
    for match in candidates:
        unchanged = text[source_cursor:match.start()]
        output_parts.append(unchanged)
        output_cursor += len(unchanged)

        source = match.group(0)
        replacement = _preserve_source_case(source, registry[source.lower()])
        output_parts.append(replacement)
        events.append(TransformationEvent(
            transform=transform,
            registry=registry_name,
            source=source,
            replacement=replacement,
            source_start=match.start(),
            source_end=match.end(),
            output_start=output_cursor,
            output_end=output_cursor + len(replacement),
        ))
        output_cursor += len(replacement)
        source_cursor = match.end()

    output_parts.append(text[source_cursor:])
    return "".join(output_parts), events


def apply_emoji_with_metadata(
    text: str,
    prob: float = 1.0,
    *,
    registry: str = "main",
    max_replacements: int | None = 2,
    rng=None,
    seed: int | None = None,
) -> tuple[str, list[TransformationEvent]]:
    """Apply principal or lossy emoji replacements with exact edit metadata."""

    local_rng = _resolve_rng(rng, seed)
    if registry == "main":
        mapping = MAIN_EMOJI_MAP
    elif registry == "lossy":
        mapping = LOSSY_EMOJI_MAP
    else:
        raise ValueError(f"Unknown emoji registry: {registry}")
    return _apply_registry(
        text,
        mapping,
        transform="emoji",
        registry_name=f"emoji_{registry}",
        probability=prob,
        max_replacements=max_replacements,
        rng=local_rng,
    )


def apply_emoji(
    text: str,
    prob: float = 1.0,
    *,
    registry: str = "main",
    max_replacements: int | None = 2,
    rng=None,
    seed: int | None = None,
) -> str:
    transformed, _ = apply_emoji_with_metadata(
        text,
        prob,
        registry=registry,
        max_replacements=max_replacements,
        rng=rng,
        seed=seed,
    )
    return transformed


def invert_from_provenance(text: str, events: Sequence[TransformationEvent | Mapping]) -> str:
    """Exactly undo non-overlapping replacements using recorded output offsets."""

    restored = text
    normalized = [event if isinstance(event, TransformationEvent) else TransformationEvent(**event)
                  for event in events]
    for event in reversed(normalized):
        actual = restored[event.output_start:event.output_end]
        if actual != event.replacement:
            raise ValueError(
                f"Cannot invert {event.transform}: expected {event.replacement!r} "
                f"at output span {event.output_start}:{event.output_end}, found {actual!r}"
            )
        restored = restored[:event.output_start] + event.source + restored[event.output_end:]
    return restored


def apply_marker_with_metadata(
    text: str,
    prob: float = 0.5,
    *,
    marker: str | None = None,
    marker_pool: str = "train",
    marker_choices: Sequence[str] | None = None,
    registry_name: str | None = None,
    placement: str = "suffix",
    rng=None,
    seed: int | None = None,
) -> tuple[str, list[TransformationEvent]]:
    """Add one marker from a named pool at a controlled sentence location."""

    if not 0.0 <= prob <= 1.0:
        raise ValueError("probability must be between 0 and 1")
    if marker_choices is None and marker_pool not in MARKER_POOLS:
        raise ValueError(f"Unknown marker pool: {marker_pool}")
    if placement not in {"suffix", "prefix"}:
        raise ValueError("placement must be 'suffix' or 'prefix'")

    local_rng = _resolve_rng(rng, seed)
    if local_rng.random() >= prob:
        return text, []
    choices = tuple(marker_choices) if marker_choices is not None else MARKER_POOLS[marker_pool]
    if not choices:
        raise ValueError("At least one marker choice is required")
    chosen = marker or local_rng.choice(choices)
    if chosen not in choices:
        raise ValueError(f"Marker {chosen!r} is not in pool {marker_pool!r}")

    if placement == "suffix":
        separator = "" if not text or text[-1].isspace() else " "
        output = f"{text}{separator}{chosen}"
        inserted = f"{separator}{chosen}"
        start = len(text)
    else:
        separator = "" if not text or text[0].isspace() else " "
        output = f"{chosen}{separator}{text}"
        inserted = f"{chosen}{separator}"
        start = 0

    event = TransformationEvent(
        transform="marker",
        registry=registry_name or f"markers_{marker_pool}",
        source="",
        replacement=inserted,
        source_start=len(text) if placement == "suffix" else 0,
        source_end=len(text) if placement == "suffix" else 0,
        output_start=start,
        output_end=start + len(inserted),
        placement=placement,
    )
    return output, [event]


def apply_psych_with_metadata(text: str) -> tuple[str, list[TransformationEvent]]:
    """Append the evaluation-only ``psych`` instruction with exact provenance.

    This helper never searches the input for the word ``psych``.  It always
    records one new hypothesis-suffix insertion, so natural occurrences remain
    distinguishable from an explicitly transformed record.
    """

    separator = "" if not text or text[-1].isspace() else " "
    inserted = f"{separator}{PSYCH_MARKER}"
    start = len(text)
    event = TransformationEvent(
        transform="psych",
        registry="psych_instruction_inversion",
        source="",
        replacement=inserted,
        source_start=start,
        source_end=start,
        output_start=start,
        output_end=start + len(inserted),
        placement="suffix",
    )
    return f"{text}{inserted}", [event]


def invert_entailment_contradiction_label(label: int) -> int:
    """Apply the frozen psych gold-label policy to one integer NLI label."""

    try:
        return PSYCH_LABEL_MAP[int(label)]
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"Psych requires an NLI label in {{0, 1, 2}}; got {label!r}") from error


def apply_noise(
    text: str,
    prob: float = 0.5,
    *,
    marker_pool: str = "train",
    placement: str = "suffix",
    rng=None,
    seed: int | None = None,
) -> str:
    transformed, _ = apply_marker_with_metadata(
        text,
        prob,
        marker_pool=marker_pool,
        placement=placement,
        rng=rng,
        seed=seed,
    )
    return transformed


def apply_slang_with_metadata(
    text: str, prob: float = 1.0, add_noise: bool = False, *, rng=None
) -> tuple[str, list[TransformationEvent]]:
    """Apply the legacy slang condition and retain stage-ordered provenance."""

    local_rng = _resolve_rng(rng)
    transformed, phrase_events = _apply_registry(
        text,
        PHRASE_SLANG,
        transform="slang",
        registry_name="slang_phrases",
        probability=prob,
        rng=local_rng,
    )
    transformed, word_events = _apply_registry(
        transformed,
        WORD_SLANG,
        transform="slang",
        registry_name="slang_words",
        probability=prob,
        rng=local_rng,
    )
    events = phrase_events + word_events
    if add_noise:
        transformed, marker_events = apply_marker_with_metadata(
            transformed, 0.3, marker_pool="train", rng=local_rng
        )
        events.extend(marker_events)
    return transformed, events


def apply_slang(text: str, prob: float = 1.0, add_noise: bool = False, *, rng=None) -> str:
    transformed, _ = apply_slang_with_metadata(text, prob, add_noise, rng=rng)
    return transformed


def transform_example(
    example: Mapping,
    mode: str,
    slang_prob: float = 1.0,
    emoji_prob: float = 1.0,
    noise_prob: float = 0.5,
    *,
    seed: int | None = None,
    rng=None,
    include_metadata: bool = False,
    emoji_registry: str = "main",
    marker_pool: str = "train",
    marker: str | None = None,
    marker_choices: Sequence[str] | None = None,
    marker_registry_name: str | None = None,
    marker_placement: str | None = None,
    max_emoji_replacements: int | None = 2,
) -> dict:
    """Transform an NLI example using a deterministic, auditable pipeline.

    Existing callers receive the original schema. Set ``include_metadata`` to add
    JSON-compatible provenance under ``transform_metadata``.
    """

    local_rng = _resolve_rng(rng, seed)
    result: MutableMapping = dict(example)
    provenance: dict[str, list[dict]] = {"premise": [], "hypothesis": []}
    remaining_emoji_replacements = max_emoji_replacements

    def emoji(field: str) -> None:
        nonlocal remaining_emoji_replacements
        transformed, events = apply_emoji_with_metadata(
            result[field], emoji_prob, registry=emoji_registry,
            max_replacements=remaining_emoji_replacements, rng=local_rng,
        )
        result[field] = transformed
        provenance[field].extend(event.to_dict() for event in events)
        if remaining_emoji_replacements is not None:
            remaining_emoji_replacements -= len(events)

    def marker_edit(field: str, placement: str = "suffix") -> None:
        transformed, events = apply_marker_with_metadata(
            result[field], noise_prob, marker=marker, marker_pool=marker_pool,
            marker_choices=marker_choices, registry_name=marker_registry_name,
            placement=placement, rng=local_rng,
        )
        result[field] = transformed
        provenance[field].extend(event.to_dict() for event in events)

    def declared_marker() -> None:
        if marker_placement is None:
            marker_edit("premise")
            marker_edit("hypothesis")
            return
        try:
            field, placement = marker_placement.rsplit("_", 1)
        except ValueError as error:
            raise ValueError(
                "marker_placement must look like 'hypothesis_suffix', "
                "'hypothesis_prefix', or 'premise_suffix'"
            ) from error
        if field not in {"premise", "hypothesis"} or placement not in {"prefix", "suffix"}:
            raise ValueError(f"Invalid marker placement: {marker_placement!r}")
        marker_edit(field, placement)

    if mode == "slang":
        for field in ("premise", "hypothesis"):
            transformed, events = apply_slang_with_metadata(
                result[field], slang_prob, rng=local_rng
            )
            result[field] = transformed
            provenance[field].extend(event.to_dict() for event in events)
    elif mode == "emoji":
        emoji("premise")
        emoji("hypothesis")
    elif mode in {"noise", "marker"}:
        declared_marker()
    elif mode in {"both", "combined"}:
        emoji("premise")
        emoji("hypothesis")
        declared_marker()
    elif mode == "lossy_emoji":
        emoji_registry = "lossy"
        emoji("premise")
        emoji("hypothesis")
    elif mode == "psych":
        transformed, events = apply_psych_with_metadata(result["hypothesis"])
        result["hypothesis"] = transformed
        provenance["hypothesis"].extend(event.to_dict() for event in events)
        result["label"] = invert_entailment_contradiction_label(result["label"])
        result["psych_applied"] = True
    else:
        raise ValueError(f"Unknown transformation mode: {mode}")

    if include_metadata:
        result["transform_metadata"] = {
            "mode": mode,
            "declared_transforms": sorted(MODE_TRANSFORMS[mode]),
            "emoji_registry": emoji_registry if "emoji" in mode or mode in {"both", "combined"} else None,
            "marker_pool": marker_pool if mode in {"noise", "marker", "both", "combined"} else None,
            "marker": marker if mode in {"noise", "marker", "both", "combined"} else None,
            "marker_placement": marker_placement if mode in {"noise", "marker", "both", "combined"} else None,
            "psych_applied": mode == "psych",
            "scope": "complete_hypothesis" if mode == "psych" else None,
            "label_policy": PSYCH_LABEL_POLICY if mode == "psych" else "preserve",
            "events": provenance,
        }
    return dict(result)


def validate_transform_condition(
    example: Mapping,
    mode: str,
    *,
    require_each_declared: bool = False,
    training: bool = False,
) -> None:
    """Validate that provenance contains only edits declared by a condition."""

    if mode not in MODE_TRANSFORMS:
        raise ValueError(f"Unknown transformation mode: {mode}")
    metadata = example.get("transform_metadata")
    if not metadata:
        raise ValueError("Condition validation requires transform_metadata")
    if metadata.get("mode") != mode:
        raise ValueError(
            f"Condition metadata declares {metadata.get('mode')!r}, expected {mode!r}"
        )
    events = [
        event
        for field_events in metadata.get("events", {}).values()
        for event in field_events
    ]
    observed = {event["transform"] for event in events}
    declared = MODE_TRANSFORMS[mode]
    unexpected = observed - declared
    if unexpected:
        raise ValueError(f"Condition {mode!r} contains undeclared transforms: {sorted(unexpected)}")
    if require_each_declared and declared - observed:
        raise ValueError(
            f"Condition {mode!r} is missing transforms: {sorted(declared - observed)}"
        )
    expected_emoji_registry = (
        "emoji_lossy" if mode == "lossy_emoji"
        else "emoji_main" if "emoji" in declared
        else None
    )
    wrong_emoji_registries = {
        event["registry"]
        for event in events
        if event["transform"] == "emoji"
        and event["registry"] != expected_emoji_registry
    }
    if wrong_emoji_registries:
        raise ValueError(
            f"Condition {mode!r} used the wrong emoji registry: "
            f"{sorted(wrong_emoji_registries)}"
        )
    if training:
        wrong_registries = {
            event["registry"]
            for event in events
            if event["transform"] == "marker" and event["registry"] != "markers_train"
        }
        if wrong_registries:
            raise ValueError(
                f"Training used non-training marker registries: {sorted(wrong_registries)}"
            )
        leaked = {
            event["replacement"].strip().lower()
            for event in events
            if event["transform"] == "marker"
            and event["replacement"].strip().lower() in HELD_OUT_MARKERS
        }
        if leaked:
            raise ValueError(f"Held-out evaluation markers entered training: {sorted(leaked)}")


def apply_training_condition(
    example: Mapping,
    *,
    augmentation_probability: float,
    marker_placements: Sequence[str],
    marker_choices: Sequence[str] | None = None,
    rng=None,
    seed: int | None = None,
    include_metadata: bool = False,
) -> dict:
    """Apply the frozen marker-augmentation condition to one training example.

    The function always returns exactly one example, so augmentation cannot alter
    the optimizer-step budget by duplicating rows.
    """

    if not marker_placements:
        raise ValueError("At least one marker placement is required")
    local_rng = _resolve_rng(rng, seed)
    placement = local_rng.choice(tuple(marker_placements))
    custom_marker_fold = marker_choices is not None
    resolved_marker_choices = tuple(marker_choices or TRAIN_MARKERS)
    transformed = transform_example(
        example,
        "noise",
        noise_prob=augmentation_probability,
        marker_pool="train",
        marker_choices=resolved_marker_choices,
        marker_registry_name=("markers_training_fold" if custom_marker_fold else "markers_train"),
        marker_placement=placement,
        rng=local_rng,
        include_metadata=True,
    )
    # The fold is validated by membership in marker_choices. The legacy
    # markers_train registry check applies only to the fixed fold-1 alias.
    events = [
        event for field_events in transformed["transform_metadata"]["events"].values()
        for event in field_events if event["transform"] == "marker"
    ]
    unexpected = {
        event["replacement"].strip() for event in events
        if event["replacement"].strip() not in resolved_marker_choices
    }
    if unexpected:
        raise ValueError(f"Training used markers outside its fold: {sorted(unexpected)}")
    if not custom_marker_fold:
        validate_transform_condition(transformed, "noise", training=True)
    if not include_metadata:
        transformed.pop("transform_metadata", None)
    return transformed
