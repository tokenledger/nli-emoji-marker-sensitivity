"""Seed-aware Phase 6 aggregation, non-inferiority, and table generation."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from transforms import PSYCH_LABEL_POLICY


APPROACH_DIRS = {
    "baseline": "baseline",
    "augmented": "augmented_{fold}",
    "clean_control": "clean_control",
    "preprocessing": "preprocessing",
    "marker_oracle": "marker_oracle",
    "hybrid": "hybrid_{fold}",
}


def _accuracy(record):
    predictions = np.asarray(record["predictions"])
    labels = np.asarray(record["labels"])
    if len(predictions) == 0 or len(predictions) != len(labels):
        raise ValueError("Malformed or empty prediction record")
    return float(np.mean(predictions == labels))


def _paired_accuracy_difference(first, second):
    first_positions = {source_id: index for index, source_id in enumerate(first["source_indices"])}
    if len(first_positions) != len(first["source_indices"]):
        raise ValueError("Duplicate source IDs in paired comparison")
    try:
        selected = [first_positions[source_id] for source_id in second["source_indices"]]
    except KeyError as error:
        raise ValueError(f"Missing paired source ID: {error.args[0]}") from error
    first_labels = [first["labels"][index] for index in selected]
    if first_labels != second["labels"]:
        raise ValueError("Labels differ in paired comparison")
    first_accuracy = np.mean([
        first["predictions"][index] == first["labels"][index] for index in selected
    ])
    return float(first_accuracy - _accuracy(second))


def seed_summary(values_by_seed):
    if len(values_by_seed) < 2:
        raise ValueError("Seed-aware uncertainty requires at least two seeds")
    ordered = sorted((int(seed), float(value)) for seed, value in values_by_seed.items())
    values = np.asarray([value for _, value in ordered])
    return {
        "seeds": [seed for seed, _ in ordered],
        "values": values.tolist(),
        "mean": float(values.mean()),
        "standard_deviation": float(values.std(ddof=1)),
        "uncertainty_scope": "training_seed_variance",
    }


def paired_seed_summary(first_by_seed, second_by_seed):
    if set(first_by_seed) != set(second_by_seed):
        raise ValueError("Paired seed comparison has different seed sets")
    differences = {
        int(seed): float(first_by_seed[seed]) - float(second_by_seed[seed])
        for seed in first_by_seed
    }
    return seed_summary(differences)


def noninferiority_test(system_by_seed, baseline_by_seed, *, margin_pp, confidence=0.95):
    summary = paired_seed_summary(system_by_seed, baseline_by_seed)
    values_pp = np.asarray(summary["values"]) * 100
    n = len(values_pp)
    standard_error = float(values_pp.std(ddof=1) / math.sqrt(n))
    try:
        from scipy.stats import t
        critical = float(t.ppf(confidence, df=n - 1))
    except ImportError as error:
        raise RuntimeError("scipy is required for the prespecified t interval") from error
    lower = float(values_pp.mean() - critical * standard_error)
    return {
        "margin_percentage_points": float(margin_pp),
        "direction": "system minus baseline",
        "confidence_level": float(confidence),
        "mean_difference_percentage_points": float(values_pp.mean()),
        "lower_one_sided_bound_percentage_points": lower,
        "decision_rule": "lower bound > -margin",
        "noninferior": bool(lower > -float(margin_pp)),
        "seeds": summary["seeds"],
        "seed_differences_percentage_points": values_pp.tolist(),
    }


def paired_example_interval(first_by_seed, second_by_seed, *, n_bootstrap=2000,
                            confidence=0.95, random_seed=42):
    if set(first_by_seed) != set(second_by_seed):
        raise ValueError("Example interval has different seed sets")
    matrices = []
    reference_ids = None
    for seed in sorted(first_by_seed):
        first = first_by_seed[seed]
        second = second_by_seed[seed]
        if first["source_indices"] != second["source_indices"]:
            raise ValueError(f"Source-ID mismatch for seed {seed}")
        if first["labels"] != second["labels"]:
            raise ValueError(f"Label mismatch for seed {seed}")
        if reference_ids is None:
            reference_ids = first["source_indices"]
        elif first["source_indices"] != reference_ids:
            raise ValueError("Source-ID mismatch across seeds")
        labels = np.asarray(first["labels"])
        matrices.append(
            (np.asarray(first["predictions"]) == labels).astype(float)
            - (np.asarray(second["predictions"]) == labels).astype(float)
        )
    per_example = np.asarray(matrices).mean(axis=0)
    rng = np.random.default_rng(random_seed)
    estimates = []
    for _ in range(n_bootstrap):
        sample = rng.integers(0, len(per_example), len(per_example))
        estimates.append(float(per_example[sample].mean()))
    alpha = 1 - confidence
    return {
        "mean_difference": float(per_example.mean()),
        "lower": float(np.quantile(estimates, alpha / 2)),
        "upper": float(np.quantile(estimates, 1 - alpha / 2)),
        "confidence_level": confidence,
        "uncertainty_scope": "paired_examples_with_seed_predictions_averaged",
        "training_seed_variance_reported_separately": True,
        "seeds": sorted(int(seed) for seed in first_by_seed),
        "examples": len(per_example),
    }


def _validated_seed_bundle(root, model, seed, role, split, fold, config_sha256):
    scope = f"{role}_{str(split).replace('/', '_').replace(':', '_')}"
    bundle = {}
    identity = None
    for approach, template in APPROACH_DIRS.items():
        directory = root / f"seed_{seed}" / f"{model}_{template.format(fold=fold)}_{scope}"
        for required in (
            "run_status.json", "run_manifest.json", "predictions_manifest.json",
            "predictions.json",
        ):
            if not (directory / required).exists():
                raise FileNotFoundError(f"Missing seed artifact: {directory / required}")
        run_manifest = json.loads((directory / "run_manifest.json").read_text())
        run_status = json.loads((directory / "run_status.json").read_text())
        if run_status.get("status") != "completed" or run_status.get("exit_code") != 0:
            raise ValueError(f"Run is not completed: {directory}")
        if run_manifest.get("config_sha256") != config_sha256:
            raise ValueError(f"Configuration mismatch in {directory}")
        if int(run_manifest.get("runtime_overrides", {}).get("seed", -1)) != int(seed):
            raise ValueError(f"Seed manifest mismatch in {directory}")
        current_identity = json.loads((directory / "predictions_manifest.json").read_text())
        if identity is None:
            identity = current_identity
        elif current_identity != identity:
            raise ValueError(f"Prediction identity mismatch in {directory}")
        bundle[approach] = json.loads((directory / "predictions.json").read_text())
    return bundle, identity


def _resolve_primary_variant(variants, candidates, description):
    for candidate in candidates:
        if candidate in variants:
            return candidate
    raise ValueError(
        f"Frozen aggregate requires {description}; expected one of {list(candidates)}"
    )


def aggregate(root, model, dataset, seeds, role, split, fold, config, config_sha256):
    bundles = {}
    identity = None
    for seed in seeds:
        bundle, current_identity = _validated_seed_bundle(
            Path(root) / dataset, model, seed, role, split, fold, config_sha256
        )
        if identity is None:
            identity = current_identity
        elif current_identity != identity:
            raise ValueError("Prediction identity differs across seeds")
        bundles[int(seed)] = bundle
    evaluated_variants = list(identity["evaluated_variants"])
    psych_variants = [
        variant for variant in evaluated_variants
        if identity.get("variant_checksums", {}).get(variant, {}).get(
            "label_policy", "preserve"
        ) == PSYCH_LABEL_POLICY
    ]
    variants = [
        "original",
        *[variant for variant in evaluated_variants if variant not in psych_variants],
    ]
    accuracies = {
        approach: {
            variant: seed_summary({
                seed: _accuracy(bundle[approach][variant])
                for seed, bundle in bundles.items()
            })
            for variant in variants
        }
        for approach in APPROACH_DIRS
    }
    baseline_clean = {
        seed: _accuracy(bundle["baseline"]["original"])
        for seed, bundle in bundles.items()
    }
    hybrid_clean = {
        seed: _accuracy(bundle["hybrid"]["original"])
        for seed, bundle in bundles.items()
    }
    emoji_variant = _resolve_primary_variant(
        variants, ("emoji_raw", "emoji"), "the bijective emoji condition"
    )
    marker_variant = _resolve_primary_variant(
        variants, (f"marker_unseen_{fold}_hypothesis_suffix",),
        f"the held-out marker condition for {fold}",
    )
    combined_candidates = [f"emoji_marker_combined_{fold}"]
    if fold == "fold_1":
        combined_candidates.insert(0, "emoji_marker_combined")
    combined_variant = _resolve_primary_variant(
        variants, combined_candidates, f"the combined condition for {fold}"
    )
    primary_specs = {
        "H1_emoji_vs_clean_baseline": (
            "baseline", "original", "baseline", emoji_variant,
        ),
        "H2_marker_vs_clean_baseline": (
            "baseline", "original", "baseline", marker_variant,
        ),
        "H3_emoji_normalization_vs_augmentation": (
            "preprocessing", emoji_variant, "augmented", emoji_variant,
        ),
        "H4_hybrid_vs_baseline_combined": (
            "hybrid", combined_variant, "baseline", combined_variant,
        ),
    }
    primary_seed_differences = {}
    for hypothesis, (approach_a, variant_a, approach_b, variant_b) in primary_specs.items():
        differences = {
            seed: _paired_accuracy_difference(
                bundle[approach_a][variant_a], bundle[approach_b][variant_b]
            )
            for seed, bundle in bundles.items()
        }
        primary_seed_differences[hypothesis] = seed_summary(differences)
    return {
        "schema_version": 1,
        "config_sha256": config_sha256,
        "prediction_identity": identity,
        "model": model, "dataset": dataset, "fold": fold,
        "accuracies": accuracies,
        "psych_instruction_inversion": {
            "gold_label_policy": PSYCH_LABEL_POLICY,
            "included_in_primary_hypothesis_family": False,
            "conditions": {
                variant: {
                    approach: seed_summary({
                        seed: _accuracy(bundle[approach][variant])
                        for seed, bundle in bundles.items()
                    })
                    for approach in APPROACH_DIRS
                }
                for variant in psych_variants
            },
        },
        "primary_seed_differences": primary_seed_differences,
        "clean_noninferiority": noninferiority_test(
            hybrid_clean, baseline_clean,
            margin_pp=config["statistics"]["clean_noninferiority_margin_percentage_points"],
            confidence=config["statistics"]["confidence_level"],
        ),
        "clean_paired_example_interval": paired_example_interval(
            {seed: bundle["hybrid"]["original"] for seed, bundle in bundles.items()},
            {seed: bundle["baseline"]["original"] for seed, bundle in bundles.items()},
        ),
    }


def latex_table(report):
    variants = list(next(iter(report["accuracies"].values())))
    lines = ["\\begin{tabular}{l" + "r" * len(variants) + "}",
             "System & " + " & ".join(variants) + " \\\\", "\\hline"]
    for approach, values in report["accuracies"].items():
        cells = [f"{values[variant]['mean'] * 100:.2f} $\\pm$ "
                 f"{values[variant]['standard_deviation'] * 100:.2f}"
                 for variant in variants]
        lines.append(approach.replace("_", "\\_") + " & " + " & ".join(cells) + " \\\\")
    lines.append("\\end{tabular}")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_root", required=True)
    parser.add_argument(
        "--model", required=True,
        help="Configured model key (for example electra, roberta, or bertweet).",
    )
    parser.add_argument("--dataset", required=True, choices=("snli", "multi_nli"))
    parser.add_argument("--eval_role", default="final", choices=("development", "final"))
    parser.add_argument("--eval_split", required=True)
    parser.add_argument("--marker_fold", default="fold_1")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    from experiment_config import config_hash, load_experiment_config
    config = load_experiment_config()
    seeds = config["models"][args.model]["seeds"]
    report = aggregate(
        args.results_root, args.model, args.dataset, seeds, args.eval_role,
        args.eval_split, args.marker_fold, config, config_hash(config),
    )
    out = Path(args.out)
    if out.exists():
        raise FileExistsError(f"Output already exists: {out}")
    from prepare_eval_sets import atomic_output_directory
    with atomic_output_directory(out) as staging:
        (staging / "aggregate_statistics.json").write_text(
            json.dumps(report, indent=2) + "\n"
        )
        (staging / "main_results.tex").write_text(latex_table(report))


if __name__ == "__main__":
    main()
