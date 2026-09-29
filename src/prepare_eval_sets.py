"""Generate fixed evaluation sets with strict transformation audits.

Every retained transformed row has provenance, differs from its clean source,
and can be exactly reconstructed.  Logical-content checksums (rather than Arrow
file checksums) make manifests stable across ``datasets`` library versions.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import os
import shutil
import tempfile
import time
from contextlib import contextmanager
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, Mapping

from experiment_config import config_hash, load_experiment_config
from helpers import load_frozen_source_split, source_dataset_identity
from preprocessing import audit_marker_removal
from transforms import (
    FORMAL_MARKER_CONTROLS,
    HELD_OUT_MARKERS,
    PSYCH_LABEL_MAP,
    PSYCH_LABEL_POLICY,
    RANDOM_PHRASE_CONTROLS,
    TRAIN_MARKERS,
    canonicalize_whitespace,
    invert_from_provenance,
    transform_example,
)


TRANSFORM_MODES = ("slang", "emoji", "noise", "combined")
AUDITABLE_TRANSFORM_MODES = TRANSFORM_MODES + ("lossy_emoji", "psych")
PRINCIPAL_MODES = frozenset(("emoji", "noise", "combined"))

PHASE3_EMOJI_CONDITIONS = (
    "emoji_raw", "emoji_gloss", "emoji_lossy_stress", "emoji_marker_combined"
)
MARKER_PLACEMENTS = (
    "hypothesis_suffix", "hypothesis_prefix", "premise_suffix"
)
EDIT_GROUPS = ("emoji", "markers", "controls", "combined", "other")
EMOJI_CONDITION_GROUPS = {
    "emoji_raw": "emoji",
    "emoji_gloss": "emoji",
    "emoji_marker_combined": "combined",
    "emoji_lossy_stress": "other",
}

@contextmanager
def atomic_output_directory(target: Path):
    """Stage a complete dataset tree and publish it with one rename.

    Existing outputs are never modified.  On failure the private staging tree is
    removed, so a failed audit cannot leave an apparently usable partial result.
    """

    target = target.resolve()
    if target.exists():
        raise FileExistsError(
            f"Output already exists: {target}. Choose a fresh --out directory; "
            "existing evaluation artifacts are never overwritten."
        )
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(
        prefix=f".{target.name}.staging-", dir=str(target.parent)
    ))
    try:
        yield staging
        os.replace(staging, target)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def _canonical_row(row: Mapping) -> str:
    return json.dumps(row, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                      default=lambda value: value.item() if hasattr(value, "item") else str(value))


def dataset_checksum(rows: Iterable[Mapping]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        digest.update(_canonical_row(row).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _text_changed(source: Mapping, transformed: Mapping) -> bool:
    return any(
        canonicalize_whitespace(source[field])
        != canonicalize_whitespace(transformed[field])
        for field in ("premise", "hypothesis")
    )


def _events(transformed: Mapping) -> list[Mapping]:
    by_field = transformed["transform_metadata"]["events"]
    return list(by_field["premise"]) + list(by_field["hypothesis"])


def _required_event_types(mode: str) -> set[str]:
    if mode == "emoji":
        return {"emoji"}
    if mode == "noise":
        return {"marker"}
    if mode == "combined":
        return {"emoji", "marker"}
    if mode == "slang":
        return {"slang"}
    if mode == "lossy_emoji":
        return {"emoji"}
    if mode == "psych":
        return {"psych"}
    raise ValueError(f"Unknown audit mode: {mode}")


def generate_audited_records(
    rows: Iterable[Mapping],
    mode: str,
    *,
    dataset_name: str,
    source_split: str,
    seed: int = 42,
    transformation_config: Mapping | None = None,
    emoji_registry: str | None = None,
    marker_pool: str | None = None,
    marker_choices: tuple[str, ...] | None = None,
    marker_by_source: Mapping[int, str] | None = None,
    marker_registry_name: str | None = None,
    marker_placement: str | None = None,
    marker_probability: float | None = None,
    condition_name: str | None = None,
    principal: bool | None = None,
    stress_test: bool = False,
) -> tuple[list[dict], dict]:
    """Transform rows and return only changed, valid examples plus an audit."""

    if mode not in AUDITABLE_TRANSFORM_MODES:
        raise ValueError(f"Unknown transformation mode: {mode}")
    source_rows = [dict(row) for row in rows]
    retained: list[dict] = []
    attempted = changed = unchanged = rejected = reconstructed = 0
    reject_reasons: Counter[str] = Counter()
    replacement_counts: Counter[str] = Counter()
    replacement_positions: Counter[str] = Counter()
    marker_frequencies: Counter[str] = Counter()
    marker_coverage_by_label: defaultdict[str, Counter[str]] = defaultdict(Counter)
    label_attempted: Counter[str] = Counter()
    label_retained: Counter[str] = Counter()
    coverage_by_label: defaultdict[str, Counter[str]] = defaultdict(Counter)
    required = _required_event_types(mode)
    settings = dict(transformation_config or {})
    emoji_probability = float(settings.get("emoji_probability", 1.0))
    resolved_marker_probability = float(
        settings.get("marker_probability", 1.0)
        if marker_probability is None else marker_probability
    )
    max_emoji_replacements = int(settings.get("max_emoji_replacements", 2))
    marker_placements = tuple(settings.get(
        "marker_placements",
        ("hypothesis_suffix", "hypothesis_prefix", "premise_suffix"),
    ))
    if not marker_placements:
        raise ValueError("Transformation config has no marker placements")
    resolved_marker_pool = marker_pool or (
        "held_out" if mode in {"noise", "combined"} else "train"
    )
    resolved_emoji_registry = emoji_registry or (
        "lossy" if mode == "lossy_emoji" else "main"
    )
    marker_counters: Counter[str] = Counter()
    emoji_replacements_per_example: Counter[str] = Counter()

    for source_index, source in enumerate(source_rows):
        attempted += 1
        label = str(source.get("label", "unknown"))
        label_attempted[label] += 1
        resolved_placement = marker_placement or marker_placements[
            (seed + source_index) % len(marker_placements)
        ]
        chosen_marker = None
        if marker_by_source is not None:
            chosen_marker = marker_by_source[source_index]
        elif marker_choices:
            # Cycle independently within each gold label. This makes every marker
            # analyzable for every NLI class instead of relying on random coverage.
            chosen_marker = marker_choices[
                (seed + marker_counters[label]) % len(marker_choices)
            ]
            marker_counters[label] += 1
        transformed = transform_example(
            source,
            mode,
            seed=seed + source_index,
            include_metadata=True,
            slang_prob=1.0,
            emoji_prob=emoji_probability,
            noise_prob=resolved_marker_probability,
            emoji_registry=resolved_emoji_registry,
            marker_pool=resolved_marker_pool,
            marker=chosen_marker,
            marker_choices=marker_choices,
            marker_registry_name=marker_registry_name,
            marker_placement=resolved_placement,
            max_emoji_replacements=max_emoji_replacements,
        )
        transformed["source_index"] = source_index
        transformed["condition"] = condition_name or mode
        events = _events(transformed)

        if not _text_changed(source, transformed):
            unchanged += 1
            reject_reasons["unchanged"] += 1
            continue
        changed += 1

        try:
            restored = {
                field: invert_from_provenance(
                    transformed[field],
                    transformed["transform_metadata"]["events"][field],
                )
                for field in ("premise", "hypothesis")
            }
        except ValueError:
            rejected += 1
            reject_reasons["provenance_inversion_error"] += 1
            continue
        if any(restored[field] != source[field] for field in restored):
            rejected += 1
            reject_reasons["inverse_mismatch"] += 1
            continue
        reconstructed += 1

        if mode == "psych":
            expected_label = PSYCH_LABEL_MAP[int(source["label"])]
            premise_events = transformed["transform_metadata"]["events"]["premise"]
            hypothesis_events = transformed["transform_metadata"]["events"]["hypothesis"]
            if transformed.get("psych_applied") is not True:
                rejected += 1
                reject_reasons["missing_psych_applied_flag"] += 1
                continue
            if premise_events or len(hypothesis_events) != 1:
                rejected += 1
                reject_reasons["invalid_psych_field_provenance"] += 1
                continue
            if int(transformed["label"]) != expected_label:
                rejected += 1
                reject_reasons["invalid_psych_label_inversion"] += 1
                continue

        observed = {event["transform"] for event in events}
        missing = required - observed
        if missing:
            rejected += 1
            reject_reasons[f"missing_required_{'_'.join(sorted(missing))}"] += 1
            continue
        emoji_count = sum(event["transform"] == "emoji" for event in events)
        if resolved_emoji_registry == "main" and emoji_count > max_emoji_replacements:
            rejected += 1
            reject_reasons["too_many_principal_emoji_replacements"] += 1
            continue
        retained.append(transformed)
        if emoji_count:
            emoji_replacements_per_example[str(emoji_count)] += 1
        label_retained[label] += 1

        for event in events:
            transform = event["transform"]
            replacement_counts[transform] += 1
            replacement_positions[f"{event.get('placement') or 'lexical'}:{event['output_start']}"] += 1
            coverage_by_label[label][transform] += 1
            if transform == "marker":
                inserted_marker = event["replacement"].strip()
                marker_frequencies[inserted_marker] += 1
                marker_coverage_by_label[inserted_marker][label] += 1

    inverse_rate = reconstructed / changed if changed else 1.0
    audit = {
        "dataset": dataset_name,
        "source_split": source_split,
        "mode": condition_name or mode,
        "transform_mode": mode,
        "principal": (mode in PRINCIPAL_MODES) if principal is None else principal,
        "stress_test": stress_test,
        "label_policy": PSYCH_LABEL_POLICY if mode == "psych" else "preserve",
        "label_changing": mode == "psych",
        "evaluation_family": (
            "psych_instruction_inversion" if mode == "psych"
            else "label_preserving_robustness"
        ),
        "resolved_transformation_settings": {
            "emoji_probability": emoji_probability,
            "marker_probability": resolved_marker_probability,
            "max_emoji_replacements": max_emoji_replacements,
            "marker_placements": list(marker_placements),
            "marker_pool": resolved_marker_pool,
            "marker_choices": list(marker_choices or ()),
            "emoji_registry": resolved_emoji_registry,
        },
        "examples": {
            "attempted": attempted,
            "changed": changed,
            "unchanged": unchanged,
            "rejected": rejected,
            "retained": len(retained),
        },
        "inverse_reconstruction": {
            "passed": reconstructed,
            "denominator_changed": changed,
            "rate": inverse_rate,
        },
        "reject_reasons": dict(sorted(reject_reasons.items())),
        "replacement_counts": dict(sorted(replacement_counts.items())),
        "emoji_replacements_per_example": dict(
            sorted(emoji_replacements_per_example.items())
        ),
        "replacement_positions": dict(sorted(replacement_positions.items())),
        "marker_frequencies": dict(sorted(marker_frequencies.items())),
        "marker_coverage_by_label": {
            marker: dict(sorted(counts.items()))
            for marker, counts in sorted(marker_coverage_by_label.items())
        },
        # This is deliberately computed from generated marker events, rather
        # than merely restating that the two registry declarations are disjoint.
        "marker_train_test_overlap": sorted(
            set(marker_frequencies) & set(TRAIN_MARKERS)
        ),
        "label_distribution": {
            "attempted": dict(sorted(label_attempted.items())),
            "retained": dict(sorted(label_retained.items())),
        },
        "transformation_coverage_by_dataset_and_label": {
            dataset_name: {
                label: dict(sorted(counts.items()))
                for label, counts in sorted(coverage_by_label.items())
            }
        },
        "checksum_sha256": dataset_checksum(retained),
    }

    if mode in {"noise", "combined"}:
        oracle_occurrences = oracle_removed = oracle_exact = 0
        global_occurrences = global_removed = global_exact = 0
        oracle_fields = 0
        for transformed in retained:
            for field in ("premise", "hypothesis"):
                marker_events = [
                    event for event in transformed["transform_metadata"]["events"][field]
                    if event["transform"] == "marker"
                ]
                if not marker_events:
                    continue
                marker_free_expected = invert_from_provenance(
                    transformed[field], marker_events
                )
                # Provenance-scoped inversion removes exactly the inserted span.
                # A global lexical deletion oracle is retained as a separate
                # diagnostic because the clean sentence may naturally contain
                # the same phrase as the inserted marker/control.
                oracle_occurrences += len(marker_events)
                oracle_removed += len(marker_events)
                roundtrip = marker_free_expected
                for event in marker_events:
                    start = int(event["output_start"])
                    roundtrip = (
                        roundtrip[:start] + event["replacement"] + roundtrip[start:]
                    )
                oracle_exact += int(roundtrip == transformed[field])
                oracle = audit_marker_removal(
                    transformed[field], marker_free_expected,
                    markers=tuple(event["replacement"].strip() for event in marker_events),
                )
                global_occurrences += oracle["marker_occurrences"]
                global_removed += oracle["marker_occurrences_removed"]
                global_exact += int(oracle["exact_reconstruction"])
                oracle_fields += 1
        audit["marker_oracle"] = {
            "fields_audited": oracle_fields,
            "marker_occurrences": oracle_occurrences,
            "marker_occurrences_removed": oracle_removed,
            "marker_removal_recall": (
                oracle_removed / oracle_occurrences if oracle_occurrences else 1.0
            ),
            "exact_reconstruction_passed": oracle_exact,
            "exact_reconstruction_rate": (
                oracle_exact / oracle_fields if oracle_fields else 1.0
            ),
        }
        audit["global_marker_deletion_diagnostic"] = {
            "fields_audited": oracle_fields,
            "lexical_occurrences": global_occurrences,
            "lexical_occurrences_removed": global_removed,
            "removal_recall": (
                global_removed / global_occurrences if global_occurrences else 1.0
            ),
            "exact_reconstruction_passed": global_exact,
            "exact_reconstruction_rate": (
                global_exact / oracle_fields if oracle_fields else 1.0
            ),
            "not_a_principal_invariant": True,
        }
    return retained, audit


def validate_principal_audit(audit: Mapping) -> None:
    """Abort generation when a retained principal set violates an invariant."""

    if not audit["principal"]:
        return
    examples = audit["examples"]
    transform_mode = audit.get("transform_mode", audit["mode"])
    if examples["retained"] == 0:
        raise ValueError(f"Principal set {audit['mode']} retained no transformed examples")
    if examples["retained"] != examples["changed"] - examples["rejected"]:
        raise ValueError(f"Principal set {audit['mode']} has inconsistent audit counts")
    if audit["inverse_reconstruction"]["rate"] != 1.0:
        raise ValueError(f"Principal set {audit['mode']} is not exactly invertible")
    if transform_mode in {"emoji", "combined"}:
        counts = {int(value) for value in audit["emoji_replacements_per_example"]}
        if not counts or min(counts) < 1 or max(counts) > 2:
            raise ValueError(
                f"Principal set {audit['mode']} must have one or two emoji replacements"
            )
    marker_status = audit.get("marker_status")
    control_type = audit.get("control_type")
    if audit["marker_train_test_overlap"] and marker_status != "seen" and not control_type:
        raise ValueError(
            "Observed evaluation markers overlap augmentation training markers: "
            f"{audit['marker_train_test_overlap']}"
        )
    if transform_mode in {"noise", "combined"} and marker_status is None:
        observed_markers = set(audit["marker_frequencies"])
        unexpected = observed_markers - set(HELD_OUT_MARKERS)
        if unexpected:
            raise ValueError(
                f"Principal marker evaluation used non-held-out markers: {sorted(unexpected)}"
            )
    marker_oracle = audit.get("marker_oracle")
    if marker_oracle and marker_oracle["marker_removal_recall"] != 1.0:
        raise ValueError(f"Principal set {audit['mode']} has incomplete marker removal")
    if marker_oracle and marker_oracle["exact_reconstruction_rate"] != 1.0:
        raise ValueError(f"Principal set {audit['mode']} fails marker-oracle reconstruction")


def paired_clean_records(source_rows: list[Mapping], transformed_rows: Iterable[Mapping]) -> list[dict]:
    """Select clean rows corresponding exactly to a transformed condition."""

    paired = []
    for transformed in transformed_rows:
        source_index = int(transformed["source_index"])
        clean = dict(source_rows[source_index])
        clean["source_index"] = source_index
        paired.append(clean)
    return paired


def generate_emoji_challenge_records(
    rows: Iterable[Mapping],
    condition: str,
    *,
    dataset_name: str,
    source_split: str,
    seed: int = 42,
    transformation_config: Mapping | None = None,
) -> tuple[list[dict], dict]:
    """Generate one of the four frozen Phase 3 emoji conditions.

    ``emoji_gloss`` deliberately renders to the original lexical text. Its rows
    retain the raw-emoji provenance, making it a paired normalization control
    rather than another perturbation.
    """

    if condition not in PHASE3_EMOJI_CONDITIONS:
        raise ValueError(f"Unknown emoji challenge condition: {condition}")
    source_rows = [dict(row) for row in rows]
    if condition == "emoji_raw":
        return generate_audited_records(
            source_rows, "emoji", dataset_name=dataset_name,
            source_split=source_split, seed=seed,
            transformation_config=transformation_config,
            condition_name=condition, principal=True,
        )
    if condition == "emoji_lossy_stress":
        return generate_audited_records(
            source_rows, "lossy_emoji", dataset_name=dataset_name,
            source_split=source_split, seed=seed,
            transformation_config=transformation_config,
            emoji_registry="lossy", condition_name=condition,
            principal=False, stress_test=True,
        )
    if condition == "emoji_marker_combined":
        return generate_audited_records(
            source_rows, "combined", dataset_name=dataset_name,
            source_split=source_split, seed=seed,
            transformation_config=transformation_config,
            marker_pool="held_out", marker_choices=tuple(HELD_OUT_MARKERS),
            marker_registry_name="markers_held_out", marker_probability=1.0,
            condition_name=condition, principal=True,
        )

    raw, raw_audit = generate_emoji_challenge_records(
        source_rows, "emoji_raw", dataset_name=dataset_name,
        source_split=source_split, seed=seed,
        transformation_config=transformation_config,
    )
    glossed: list[dict] = []
    for raw_row in raw:
        result = dict(raw_row)
        raw_events = result["transform_metadata"]["events"]
        for field in ("premise", "hypothesis"):
            result[field] = invert_from_provenance(result[field], raw_events[field])
        result["condition"] = condition
        result["transform_metadata"] = {
            **result["transform_metadata"],
            "mode": "emoji_gloss",
            "declared_transforms": ["emoji_gloss"],
            "raw_emoji_events": raw_events,
            "events": {"premise": [], "hypothesis": []},
            "renders_clean_lexical_content": True,
        }
        glossed.append(result)
    audit = {
        **raw_audit,
        "mode": condition,
        "transform_mode": "emoji_gloss",
        "principal": False,
        "control_type": "textual_emoji_gloss",
        "output_equals_paired_clean": True,
        "checksum_sha256": dataset_checksum(glossed),
    }
    return glossed, audit


def _balanced_marker_assignment(
    rows: list[Mapping], markers: tuple[str, ...], seed: int
) -> dict[int, str]:
    counters: Counter[str] = Counter()
    assignment: dict[int, str] = {}
    for index, row in enumerate(rows):
        label = str(row.get("label", "unknown"))
        assignment[index] = markers[(seed + counters[label]) % len(markers)]
        counters[label] += 1
    return assignment


def _length_matched_assignment(
    reference: Mapping[int, str], controls: tuple[str, ...], seed: int
) -> dict[int, str]:
    by_length: defaultdict[int, list[str]] = defaultdict(list)
    for control in controls:
        by_length[len(control.split())].append(control)
    counters: Counter[int] = Counter()
    matched: dict[int, str] = {}
    for index, marker in reference.items():
        length = len(marker.split())
        candidates = by_length.get(length, [])
        if not candidates:
            raise ValueError(f"No {length}-token control matches marker {marker!r}")
        matched[index] = candidates[(seed + counters[length]) % len(candidates)]
        counters[length] += 1
    return matched


def generate_marker_challenge_records(
    rows: Iterable[Mapping],
    *,
    dataset_name: str,
    source_split: str,
    fold: Mapping,
    status: str,
    placement: str,
    seed: int = 42,
    control_type: str | None = None,
    transformation_config: Mapping | None = None,
    minimum_per_label: int = 25,
) -> tuple[list[dict], dict]:
    """Generate a balanced seen/unseen marker condition or matched control."""

    if status not in {"seen", "unseen"}:
        raise ValueError("status must be 'seen' or 'unseen'")
    if placement not in MARKER_PLACEMENTS:
        raise ValueError(f"Unknown marker placement: {placement}")
    if control_type not in {None, "formal", "random"}:
        raise ValueError("control_type must be None, 'formal', or 'random'")
    source_rows = [dict(row) for row in rows]
    reference_markers = tuple(fold["train"] if status == "seen" else fold["test"])
    reference = _balanced_marker_assignment(source_rows, reference_markers, seed)
    if control_type == "formal":
        pool = tuple(FORMAL_MARKER_CONTROLS)
        assignment = _length_matched_assignment(reference, pool, seed)
    elif control_type == "random":
        pool = tuple(RANDOM_PHRASE_CONTROLS)
        assignment = _length_matched_assignment(reference, pool, seed)
    else:
        pool = reference_markers
        assignment = reference
    name = f"marker_{status}_{fold['id']}_{placement}"
    if control_type:
        name += f"_{control_type}_control"
    records, audit = generate_audited_records(
        source_rows, "noise", dataset_name=dataset_name,
        source_split=source_split, seed=seed,
        transformation_config=transformation_config,
        marker_pool=status, marker_choices=pool, marker_by_source=assignment,
        marker_registry_name=f"markers_{control_type or status}_{fold['id']}",
        marker_placement=placement, marker_probability=1.0,
        condition_name=name, principal=True,
    )
    audit.update({
        "marker_fold": fold["id"],
        "marker_status": status,
        "placement": placement,
        "control_type": control_type,
        "reference_marker_by_source": {
            str(index): marker for index, marker in reference.items()
        } if control_type else None,
        "token_count_matched": bool(control_type),
    })
    required_markers = set(assignment.values())
    missing = {
        (marker, label)
        for marker in required_markers
        for label in ("0", "1", "2")
        if audit["marker_coverage_by_label"].get(marker, {}).get(label, 0)
        < minimum_per_label
    }
    if missing:
        marker, label = next(iter(missing))
        raise ValueError(
            f"Condition {name} lacks required coverage for marker {marker!r}, "
            f"label {label}; need {minimum_per_label}"
        )
    # Overlap is expected for the explicitly seen condition, but forbidden for
    # the generalization result.
    if status == "unseen" and not control_type:
        leaked = set(audit["marker_frequencies"]) & set(fold["train"])
        if leaked:
            raise ValueError(f"Held-out marker leakage in {name}: {sorted(leaked)}")
    audit["global_registry_train_overlap"] = audit["marker_train_test_overlap"]
    audit["marker_train_test_overlap"] = sorted(
        set(audit["marker_frequencies"]) & set(fold["train"])
    ) if not control_type else []
    validate_principal_audit(audit)
    return records, audit


def generate_fold_combined_records(
    rows: Iterable[Mapping], *, dataset_name: str, source_split: str,
    fold: Mapping, seed: int = 42, transformation_config: Mapping | None = None,
) -> tuple[list[dict], dict]:
    """Generate emoji plus markers held out for one specific training fold."""

    name = f"emoji_marker_combined_{fold['id']}"
    records, audit = generate_audited_records(
        rows, "combined", dataset_name=dataset_name, source_split=source_split,
        seed=seed, transformation_config=transformation_config,
        marker_pool="unseen", marker_choices=tuple(fold["test"]),
        marker_registry_name=f"markers_unseen_{fold['id']}",
        marker_probability=1.0, condition_name=name, principal=True,
    )
    audit.update({
        "marker_fold": fold["id"],
        "marker_status": "unseen",
        "global_registry_train_overlap": audit["marker_train_test_overlap"],
        "marker_train_test_overlap": sorted(
            set(audit["marker_frequencies"]) & set(fold["train"])
        ),
    })
    validate_principal_audit(audit)
    return records, audit


def stratified_holdout(rows: Iterable[Mapping], fraction: float, seed: int) -> tuple[list[dict], list[dict]]:
    """Deterministically split rows by label without consulting final sets."""

    if not 0.0 < fraction < 1.0:
        raise ValueError("fraction must be between zero and one")
    import random

    grouped: defaultdict[int, list[dict]] = defaultdict(list)
    for index, row in enumerate(rows):
        item = dict(row)
        item["source_index"] = index
        grouped[int(item["label"])].append(item)
    development: list[dict] = []
    training: list[dict] = []
    for label, label_rows in sorted(grouped.items()):
        local = list(label_rows)
        random.Random(seed + label).shuffle(local)
        count = max(1, round(len(local) * fraction))
        development.extend(local[:count])
        training.extend(local[count:])
    development.sort(key=lambda row: row["source_index"])
    training.sort(key=lambda row: row["source_index"])
    return training, development


def resolve_worker_budget(requested: int) -> int:
    if requested < 0:
        raise ValueError('workers must be zero (auto) or a positive integer')
    available = max(1, os.cpu_count() or 1)
    return available if requested == 0 else requested


def resolve_worker_count(requested: int, task_count: int) -> int:
    return max(1, min(resolve_worker_budget(requested), max(1, task_count)))


def distribute_worker_budget(total: int, slots: int) -> list[int]:
    """Distribute a fixed CPU budget deterministically across active splits."""

    if total < 1 or slots < 1 or total < slots:
        raise ValueError('worker budget must cover every active slot')
    base, remainder = divmod(total, slots)
    return [base + int(index < remainder) for index in range(slots)]


_CONDITION_CONTEXT: dict = {}


def _initialize_condition_worker(context: dict) -> None:
    global _CONDITION_CONTEXT
    _CONDITION_CONTEXT = context


def _generate_condition_artifact(spec: dict) -> dict:
    """Generate and save one independent condition for the active split."""

    import datasets

    context = _CONDITION_CONTEXT
    source_rows = context['source_rows']
    dataset_name = context['dataset_name']
    split_name = context['split_name']
    seed = context['seed']
    transformations = context['transformations']
    kind = spec['kind']
    if kind == 'slang':
        records, audit = generate_audited_records(
            source_rows, 'slang', dataset_name=dataset_name,
            source_split=split_name, seed=seed,
            transformation_config=transformations,
            condition_name='slang', principal=False,
        )
        name = 'slang'
    elif kind == 'psych':
        name = 'psych'
        records, audit = generate_audited_records(
            source_rows, 'psych', dataset_name=dataset_name,
            source_split=split_name, seed=seed,
            transformation_config=transformations,
            condition_name=name, principal=False,
        )
    elif kind == 'emoji':
        name = spec['condition']
        records, audit = generate_emoji_challenge_records(
            source_rows, name, dataset_name=dataset_name,
            source_split=split_name, seed=seed,
            transformation_config=transformations,
        )
        if audit['principal']:
            validate_principal_audit(audit)
    elif kind == 'fold_combined':
        fold = spec['fold']
        name = f"emoji_marker_combined_{fold['id']}"
        records, audit = generate_fold_combined_records(
            source_rows, dataset_name=dataset_name,
            source_split=split_name, fold=fold, seed=seed,
            transformation_config=transformations,
        )
    elif kind == 'marker':
        records, audit = generate_marker_challenge_records(
            source_rows, dataset_name=dataset_name,
            source_split=split_name, fold=spec['fold'],
            status=spec['status'], placement=spec['placement'], seed=seed,
            control_type=spec.get('control_type'),
            transformation_config=transformations,
        )
        name = audit['mode']
    else:
        raise ValueError(f'Unknown condition task kind: {kind!r}')

    variant = None
    if records:
        split_path = Path(context['split_path'])
        datasets.Dataset.from_list(records).save_to_disk(str(split_path / name))
        clean_records = paired_clean_records(source_rows, records)
        relative = Path(dataset_name) / context['split_role'] / split_name
        variant = {
            'path': str(relative / name),
            'paired_clean_storage': 'source_index_view',
            'examples': len(records),
            'checksum_sha256': audit['checksum_sha256'],
            'paired_clean_checksum_sha256': dataset_checksum(clean_records),
            'label_policy': audit['label_policy'],
            'label_changing': audit['label_changing'],
            'evaluation_family': audit['evaluation_family'],
        }
    return {
        'id': spec['id'],
        'name': name,
        'audit': audit,
        'variant': variant,
        'primary_noise': bool(spec.get('primary_noise')),
    }


def _condition_specs(config: Mapping) -> list[dict]:
    specs: list[dict] = [{'id': 'slang', 'kind': 'slang'}]
    if config['transformations'].get('psych'):
        specs.append({'id': 'psych', 'kind': 'psych'})
    specs.extend(
        {'id': f'emoji:{condition}', 'kind': 'emoji', 'condition': condition}
        for condition in PHASE3_EMOJI_CONDITIONS
    )
    for fold in config['transformations']['marker_folds']:
        if fold['id'] != 'fold_1':
            specs.append({
                'id': f"combined:{fold['id']}",
                'kind': 'fold_combined', 'fold': fold,
            })
        for status in ('seen', 'unseen'):
            for placement in MARKER_PLACEMENTS:
                for control_type in (None, 'formal', 'random'):
                    specs.append({
                        'id': (
                            f"marker:{fold['id']}:{status}:{placement}:"
                            f"{control_type or 'principal'}"
                        ),
                        'kind': 'marker',
                        'fold': fold,
                        'status': status,
                        'placement': placement,
                        'control_type': control_type,
                        'primary_noise': (
                            fold['id'] == 'fold_1'
                            and status == 'unseen'
                            and placement == 'hypothesis_suffix'
                            and control_type is None
                        ),
                    })
    return specs


def condition_edit_group(spec: Mapping) -> str:
    """Name the group of edits that a generated condition belongs to."""

    kind = spec['kind']
    if kind == 'emoji':
        return EMOJI_CONDITION_GROUPS[spec['condition']]
    if kind == 'fold_combined':
        return 'combined'
    if kind == 'marker':
        return 'controls' if spec['control_type'] else 'markers'
    return 'other'


def _run_condition_tasks(context: dict, specs: list[dict], workers: int) -> list[dict]:
    workers = max(1, min(workers, len(specs)))
    _initialize_condition_worker(context)
    if workers == 1:
        return [_generate_condition_artifact(spec) for spec in specs]

    completed: dict[str, dict] = {}
    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=_initialize_condition_worker,
        initargs=(context,),
    ) as executor:
        futures = {
            executor.submit(_generate_condition_artifact, spec): spec
            for spec in specs
        }
        try:
            for index, future in enumerate(as_completed(futures), 1):
                result = future.result()
                completed[result['id']] = result
                if index == 1 or index % 10 == 0 or index == len(specs):
                    print(
                        f"[conditions:{context['dataset_name']}/"
                        f"{context['split_name']}] {index}/{len(specs)}",
                        flush=True,
                    )
        except BaseException:
            for future in futures:
                future.cancel()
            raise
    return [completed[spec['id']] for spec in specs]


def _generate_split_artifacts(task: dict, source_dataset) -> tuple[str, str, dict]:
    """Generate one independent dataset/split tree and its manifest entry."""

    dataset_name = task['dataset_name']
    split_role = task['split_role']
    split_name = task['split_name']
    config = task['config']
    seed = int(task['seed'])
    if not task.get('source_already_filtered', False):
        source_dataset = source_dataset.filter(lambda ex: ex['label'] != -1)
    source_rows = [dict(row) for row in source_dataset]
    split_path = Path(task['staging_dir']) / dataset_name / split_role / split_name
    split_path.mkdir(parents=True, exist_ok=True)
    source_dataset.save_to_disk(str(split_path / 'original'))
    print(
        f'[eval:{dataset_name}/{split_role}:{split_name}] '
        f'{len(source_rows):,} source examples', flush=True,
    )
    specs = _condition_specs(config)
    edits = task.get('edits')
    if edits:
        specs = [spec for spec in specs if condition_edit_group(spec) in edits]
    condition_context = {
        'source_rows': source_rows,
        'split_path': str(split_path),
        'dataset_name': dataset_name,
        'split_role': split_role,
        'split_name': split_name,
        'seed': seed,
        'transformations': config['transformations'],
    }
    generated = _run_condition_tasks(
        condition_context, specs, int(task.get('condition_workers', 1))
    )
    audits: dict[str, dict] = {}
    variants: dict[str, dict] = {}
    primary_noise_name: str | None = None
    for result in generated:
        audits[result['name']] = result['audit']
        if result['variant'] is not None:
            variants[result['name']] = result['variant']
        if result['primary_noise']:
            primary_noise_name = result['name']

    aliases = {'emoji': 'emoji_raw', 'combined': 'emoji_marker_combined'}
    for fold in config['transformations']['marker_folds']:
        fold_combined_name = f"emoji_marker_combined_{fold['id']}"
        if fold['id'] == 'fold_1' and 'emoji_marker_combined' in variants:
            variants[fold_combined_name] = {
                **variants['emoji_marker_combined'],
                'alias_of': 'emoji_marker_combined',
            }
    if primary_noise_name is not None:
        aliases['noise'] = primary_noise_name
    elif not edits or 'markers' in edits:
        raise AssertionError('Frozen fold_1 primary unseen condition was not generated')
    aliases = {
        alias: target_name for alias, target_name in aliases.items()
        if target_name in variants
    }
    for alias, target_name in aliases.items():
        variants[alias] = {**variants[target_name], 'alias_of': target_name}

    (split_path / 'audit_report.json').write_text(
        json.dumps(audits, indent=2, ensure_ascii=False) + '\n',
        encoding='utf-8',
    )
    split_key = f'{split_role}:{split_name}'
    split_entry = {
        'role': split_role,
        'source_split': split_name,
        'labels_access': (
            'tuning_allowed' if split_role == 'development'
            else 'final_evaluation_only'
        ),
        'generated_after_config_freeze': True,
        'source_examples': len(source_rows),
        'source_checksum_sha256': dataset_checksum(source_rows),
        'base_path': str(Path(dataset_name) / split_role / split_name),
        'variants': variants,
        'aliases': aliases,
    }
    return dataset_name, split_key, split_entry


def _generate_split_task(task: dict) -> tuple[str, str, dict]:
    source_dataset = load_frozen_source_split(
        task['source_snapshot'], task['dataset_name'],
        task['split_role'], task['split_name'],
    )
    return _generate_split_artifacts(task, source_dataset)


def _run_split_tasks(tasks: list[dict], workers: int) -> list[tuple[str, str, dict]]:
    worker_budget = resolve_worker_budget(workers)
    resolved_workers = min(worker_budget, len(tasks))
    allocations = distribute_worker_budget(worker_budget, resolved_workers)
    tasks = [dict(task) for task in tasks]
    for index, task in enumerate(tasks):
        task['condition_workers'] = (
            allocations[index] if index < resolved_workers else 1
        )
    print(
        f'Generating {len(tasks)} independent evaluation splits with '
        f'{resolved_workers} split process(es) and a {worker_budget}-CPU '
        f'condition budget...', flush=True,
    )
    started = time.monotonic()
    results: dict[tuple[str, str], tuple[str, str, dict]] = {}
    if resolved_workers == 1:
        for index, task in enumerate(tasks, 1):
            result = _generate_split_task(task)
            results[(result[0], result[1])] = result
            print(
                f'[eval {index}/{len(tasks)}] completed {result[0]}/{result[1]} '
                f'({(time.monotonic() - started) / 60:.1f} min)', flush=True,
            )
    else:
        with ProcessPoolExecutor(max_workers=resolved_workers) as executor:
            future_tasks = {
                executor.submit(_generate_split_task, task): task for task in tasks
            }
            try:
                for index, future in enumerate(as_completed(future_tasks), 1):
                    result = future.result()
                    results[(result[0], result[1])] = result
                    print(
                        f'[eval {index}/{len(tasks)}] completed '
                        f'{result[0]}/{result[1]} '
                        f'({(time.monotonic() - started) / 60:.1f} min)', flush=True,
                    )
            except BaseException:
                for future in future_tasks:
                    future.cancel()
                raise
    return [
        results[(task['dataset_name'], f"{task['split_role']}:{task['split_name']}")]
        for task in tasks
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=str, default='./eval_sets',
                        help='Directory to save fixed eval sets')
    parser.add_argument('--config', type=str, default=None,
                        help='Frozen experiment configuration (default: experiment_v6.json)')
    parser.add_argument('--seed', type=int, default=None,
                        help='Must match the frozen transformation generation seed')
    parser.add_argument(
        '--source_snapshot', type=str, default=None,
        help=('Immutable release from prepare_data_release.py. Production generation '
              'must use this instead of downloading datasets again.'),
    )
    parser.add_argument(
        '--workers', type=int, default=0,
        help='Total parallel CPU budget across splits and conditions; 0 uses all CPUs.',
    )
    parser.add_argument(
        '--datasets', nargs='+', default=None, choices=['snli', 'multi_nli'],
        help='Generate only these datasets (default: both).',
    )
    parser.add_argument(
        '--splits', nargs='+', default=None,
        help='Generate only these source splits, for example test or '
             'validation_matched (default: every development and final split).',
    )
    parser.add_argument(
        '--edits', nargs='+', default=None, choices=list(EDIT_GROUPS),
        help='Generate only these groups of conditions (default: all groups).',
    )
    args = parser.parse_args()

    import datasets

    if args.workers < 0:
        parser.error('--workers must be zero (auto) or a positive integer')
    config = load_experiment_config(args.config)
    frozen_seed = int(config['transformations']['generation_seed'])
    if args.seed is not None and args.seed != frozen_seed:
        raise ValueError(
            f'Principal generation seed is frozen at {frozen_seed}; got {args.seed}'
        )
    args.seed = frozen_seed
    config_sha = config_hash(config)
    out_dir = Path(args.out)
    manifest = {
        'schema_version': 4,
        'experiment_id': config['experiment_id'],
        'config_sha256': config_sha,
        'seed': args.seed,
        'final_labels_policy': (
            'Final labels may be loaded only by final-evaluation commands; '
            'development/tuning code must use development-role suites.'
        ),
        'label_policies': {
            'preserve': {
                'label_changing': False,
                'validation': 'transformed label must equal paired-clean label',
            },
            PSYCH_LABEL_POLICY: {
                'label_changing': True,
                'mapping': {'0': 2, '1': 1, '2': 0},
                'validation': (
                    'requires psych_applied=True and exact hypothesis-suffix provenance'
                ),
            },
        },
        'datasets': {},
    }
    if args.datasets or args.splits or args.edits:
        manifest['generation_filter'] = {
            'datasets': sorted(args.datasets or []),
            'splits': sorted(args.splits or []),
            'edits': sorted(args.edits or []),
        }
    if args.source_snapshot:
        manifest['source_manifest_sha256'] = source_dataset_identity(
            args.source_snapshot, 'snli'
        )['source_manifest_sha256']

    with atomic_output_directory(out_dir) as staging_dir:
        tasks: list[dict] = []
        live_jobs: list[tuple[dict, object]] = []
        for dataset_name in (args.datasets or ('snli', 'multi_nli')):
            hub_name = config['dataset_protocol'][dataset_name]['huggingface_name']
            protocol = config['dataset_protocol'][dataset_name]
            if args.source_snapshot:
                identity = source_dataset_identity(args.source_snapshot, dataset_name)
                if identity['source_config_sha256'] != config_sha:
                    raise ValueError(
                        f'Source snapshot/config mismatch for {dataset_name}: '
                        f'{identity["source_config_sha256"]} != {config_sha}'
                    )
                if identity['huggingface_repository'] != hub_name:
                    raise ValueError(
                        f'Source repository mismatch for {dataset_name}: '
                        f'{identity["huggingface_repository"]} != {hub_name}'
                    )
                development_name = (
                    protocol.get('development_split')
                    if dataset_name == 'snli' else 'train_holdout'
                )
                split_definitions = [
                    ('development', development_name),
                    *[('final', name) for name in protocol['final_splits']],
                ]
                source_revision = identity['huggingface_revision']
            else:
                print(
                    'WARNING: downloading live Hub data. Use --source_snapshot for '
                    'paper-grade generation.', flush=True,
                )
                raw = datasets.load_dataset(hub_name)
                source_revision = None
                if dataset_name == 'snli':
                    live_sources = [
                        ('development', 'validation', raw['validation']),
                        ('final', 'test', raw['test']),
                    ]
                else:
                    filtered_train = raw['train'].filter(lambda ex: ex['label'] != -1)
                    development_split = filtered_train.train_test_split(
                        test_size=protocol['development_fraction'],
                        seed=protocol['development_seed'],
                        stratify_by_column='label',
                    )
                    live_sources = [
                        ('development', 'train_holdout', development_split['test']),
                        ('final', 'validation_matched', raw['validation_matched']),
                        ('final', 'validation_mismatched', raw['validation_mismatched']),
                    ]
                split_definitions = [(role, name) for role, name, _ in live_sources]

            manifest['datasets'][dataset_name] = {
                'huggingface_name': hub_name,
                'huggingface_revision': source_revision,
                'default_development_split': split_definitions[0][1],
                'default_final_split': protocol['final_splits'][0],
                'splits': {},
            }
            if args.splits:
                split_definitions = [
                    (role, name) for role, name in split_definitions
                    if name in args.splits
                ]
            for split_role, split_name in split_definitions:
                task = {
                    'source_snapshot': args.source_snapshot,
                    'staging_dir': str(staging_dir),
                    'dataset_name': dataset_name,
                    'split_role': split_role,
                    'split_name': split_name,
                    'seed': args.seed,
                    'config': config,
                    'source_already_filtered': bool(args.source_snapshot),
                    'edits': args.edits,
                    'condition_workers': resolve_worker_budget(args.workers),
                }
                if args.source_snapshot:
                    tasks.append(task)
                else:
                    source_dataset = next(
                        source for role, name, source in live_sources
                        if role == split_role and name == split_name
                    )
                    live_jobs.append((task, source_dataset))

        if not tasks and not live_jobs:
            raise SystemExit(
                f'No split matches --datasets {args.datasets} --splits {args.splits}'
            )
        if args.source_snapshot:
            generated = _run_split_tasks(tasks, args.workers)
        else:
            print('Live-Hub compatibility mode uses one process.', flush=True)
            generated = [
                _generate_split_artifacts(task, source_dataset)
                for task, source_dataset in live_jobs
            ]
        for dataset_name, split_key, split_entry in generated:
            manifest['datasets'][dataset_name]['splits'][split_key] = split_entry

        (staging_dir / 'dataset_manifest.json').write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False) + '\n', encoding='utf-8'
        )
    print(f'\nAll audited eval sets and manifest saved to {out_dir}/')


if __name__ == "__main__":
    main()
