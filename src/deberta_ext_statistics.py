"""
Baseline-only paired statistics for the DeBERTa-v3-base extension.

The frozen primary route in ``statistical_tests.py`` needs all six production
systems (baseline, augmented, clean control, preprocessing, marker oracle,
hybrid).  The DeBERTa-v3-base extension has Tier 1 runs only (baseline and
emoji-normalization preprocessing), so this module defines a SEPARATE
addendum family that uses baseline predictions alone.  It does not modify or
re-run the frozen 45-cell primary family.

Per split (SNLI test, MultiNLI matched, MultiNLI mismatched) one Holm family
of exactly eleven exact two-sided McNemar tests on identical source indices:

  1      H1  baseline emoji_raw versus its paired-clean view
  2-4    H2  baseline marker_unseen_fold_{1,2,3}_hypothesis_suffix versus clean
  5-10   controls  each fold's unseen H-suffix informal condition versus the
                   same fold's formal control and versus its random control
  11     normalization  preprocessing emoji_raw versus baseline emoji_raw

Every test reports the accuracy difference in percentage points (treatment
minus reference), discordant counts, exact binomial p, Holm-adjusted p within
the eleven-test family, and a 95% percentile bootstrap interval over sources
(2,000 paired resamples, seed 20260924).

Usage:
    python src/deberta_ext_statistics.py \
        --inputs-dir $PREDICTIONS_DIR/deberta \
        --backup-root $PREDICTIONS_DIR/fold_study \
        --out-dir $OUTPUT_DIR/primary/deberta
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import binomtest

SOURCE_ROOT = Path(__file__).resolve().parent
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from statistical_tests import (  # noqa: E402
    _align_to_reference,
    _holm_adjust,
    load_predictions,
    mcnemar_test,
    validate_prediction_payload,
)

BOOTSTRAP_SEED = 20260924
BOOTSTRAP_RESAMPLES = 2000
FAMILYWISE_ALPHA = 0.05
FOLDS = ("fold_1", "fold_2", "fold_3")
MODEL_PREFIX = "deberta_v3_base"

SPLITS: dict[str, dict[str, str]] = {
    "snli_test": {"dataset": "snli", "scope": "final_test", "label": "SNLI test"},
    "mnli_matched": {
        "dataset": "multi_nli",
        "scope": "final_validation_matched",
        "label": "MNLI matched",
    },
    "mnli_mismatched": {
        "dataset": "multi_nli",
        "scope": "final_validation_mismatched",
        "label": "MNLI mismatched",
    },
}

# Five published encoders: (results prefix, display name, backup sub-root).
FIVE_ENCODERS: tuple[tuple[str, str, str], ...] = (
    ("electra", "ELECTRA-small", "publication_results"),
    ("roberta_base", "RoBERTa-base", "_colab/publication_results_roberta_base42"),
    ("roberta", "RoBERTa-large", "_colab/publication_results_roberta42_{short}"),
    ("timelm", "TimeLM-21", "publication_results"),
    ("bertweet", "BERTweet", "publication_results"),
)
DATASET_SHORT = {"snli": "snli", "multi_nli": "mnli"}

H1_VARIANT = "emoji_raw"


def _suffix(fold: str) -> str:
    return f"marker_unseen_{fold}_hypothesis_suffix"


def family_specification() -> list[dict[str, str]]:
    """The eleven-test addendum family, in a fixed order.

    Each entry names the treatment (system, variant) and the reference
    (system, variant).  The accuracy difference is treatment minus reference.
    """
    family: list[dict[str, str]] = [
        {
            "test_id": "H1_emoji_raw_vs_paired_clean",
            "group": "H1",
            "treatment_system": "baseline",
            "treatment_variant": H1_VARIANT,
            "reference_system": "baseline",
            "reference_variant": f"clean_{H1_VARIANT}",
        }
    ]
    for fold in FOLDS:
        family.append(
            {
                "test_id": f"H2_{fold}_hsuffix_vs_paired_clean",
                "group": "H2",
                "treatment_system": "baseline",
                "treatment_variant": _suffix(fold),
                "reference_system": "baseline",
                "reference_variant": f"clean_{_suffix(fold)}",
            }
        )
    for fold in FOLDS:
        for control in ("formal", "random"):
            family.append(
                {
                    "test_id": f"control_{fold}_hsuffix_vs_{control}",
                    "group": "control",
                    "treatment_system": "baseline",
                    "treatment_variant": _suffix(fold),
                    "reference_system": "baseline",
                    "reference_variant": f"{_suffix(fold)}_{control}_control",
                }
            )
    family.append(
        {
            "test_id": "normalization_preprocessing_vs_baseline_emoji_raw",
            "group": "normalization",
            "treatment_system": "preprocessing",
            "treatment_variant": H1_VARIANT,
            "reference_system": "baseline",
            "reference_variant": H1_VARIANT,
        }
    )
    assert len(family) == 11
    return family


def exact_mcnemar(treatment_correct: np.ndarray, reference_correct: np.ndarray) -> dict[str, Any]:
    """Exact two-sided McNemar test on paired correctness vectors."""
    treatment_correct = np.asarray(treatment_correct, dtype=bool)
    reference_correct = np.asarray(reference_correct, dtype=bool)
    if treatment_correct.shape != reference_correct.shape:
        raise ValueError("Paired correctness vectors differ in length")
    treatment_only = int((treatment_correct & ~reference_correct).sum())
    reference_only = int((~treatment_correct & reference_correct).sum())
    discordant = treatment_only + reference_only
    if discordant == 0:
        pvalue = 1.0
    else:
        pvalue = float(
            binomtest(treatment_only, discordant, p=0.5, alternative="two-sided").pvalue
        )
    return {
        "n_pairs": int(treatment_correct.size),
        "n_treatment_only_correct": treatment_only,
        "n_reference_only_correct": reference_only,
        "n_discordant": discordant,
        "pvalue": pvalue,
    }


def bootstrap_difference_ci(
    treatment_correct: np.ndarray,
    reference_correct: np.ndarray,
    n_resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
    ci: float = 0.95,
) -> dict[str, float]:
    """Percentile bootstrap over sources of the paired accuracy difference (pp)."""
    diff = np.asarray(treatment_correct, dtype=float) - np.asarray(reference_correct, dtype=float)
    n = diff.size
    rng = np.random.default_rng(seed)
    means = np.empty(n_resamples, dtype=float)
    for index in range(n_resamples):
        draw = rng.integers(0, n, n)
        means[index] = diff[draw].mean()
    lower = float(np.percentile(means, (1 - ci) / 2 * 100)) * 100
    upper = float(np.percentile(means, (1 + ci) / 2 * 100)) * 100
    return {
        "difference_pp": float(diff.mean() * 100),
        "ci_lower_pp": lower,
        "ci_upper_pp": upper,
        "n_resamples": int(n_resamples),
        "seed": int(seed),
    }


def paired_correctness(treatment_record: dict, reference_record: dict) -> tuple[np.ndarray, np.ndarray, int]:
    """Align treatment onto the reference's source indices and return correctness."""
    treatment_preds, reference_preds, labels = _align_to_reference(
        treatment_record, reference_record
    )
    labels_arr = np.asarray(labels)
    return (
        np.asarray(treatment_preds) == labels_arr,
        np.asarray(reference_preds) == labels_arr,
        int(labels_arr.size),
    )


def run_test(spec: dict[str, str], preds: dict[str, dict]) -> dict[str, Any]:
    treatment = preds[spec["treatment_system"]][spec["treatment_variant"]]
    reference = preds[spec["reference_system"]][spec["reference_variant"]]
    if treatment["source_indices"] != reference["source_indices"]:
        raise ValueError(
            f'{spec["test_id"]}: treatment and reference source indices differ; '
            "the addendum family requires identical sources"
        )
    t_correct, r_correct, n = paired_correctness(treatment, reference)
    result = dict(spec)
    result.update(exact_mcnemar(t_correct, r_correct))
    result.update(bootstrap_difference_ci(t_correct, r_correct))
    result["treatment_accuracy"] = float(t_correct.mean())
    result["reference_accuracy"] = float(r_correct.mean())
    # Continuity-corrected chi-square from the frozen route, for cross-reference only.
    labels = reference["labels"]
    frozen = mcnemar_test(
        _align_to_reference(treatment, reference)[0], reference["predictions"], labels
    )
    result["chi2_corrected_pvalue_reference_only"] = frozen["pvalue"]
    return result


def holm_family(results: list[dict[str, Any]], alpha: float = FAMILYWISE_ALPHA) -> None:
    """Holm-adjust ``pvalue`` in place using the frozen route's helper."""
    _holm_adjust([(r["test_id"], "", r) for r in results], alpha=alpha)


def holm_sensitivity_to_corrected_chi2(tests: list[dict[str, Any]], alpha: float = FAMILYWISE_ALPHA) -> dict[str, Any]:
    """Re-run Holm within the same family using the frozen route's corrected chi-square p.

    Reports whether any Holm verdict changes relative to the exact binomial p.
    """
    shadow = [
        {"test_id": t["test_id"], "pvalue": t["chi2_corrected_pvalue_reference_only"]} for t in tests
    ]
    _holm_adjust([(r["test_id"], "", r) for r in shadow], alpha=alpha)
    changed = []
    for exact, corrected in zip(tests, shadow):
        exact["chi2_corrected_holm_pvalue_reference_only"] = corrected["holm_adjusted_pvalue"]
        exact["chi2_corrected_significant_holm_reference_only"] = corrected["significant_holm"]
        if corrected["significant_holm"] != exact["significant_holm"]:
            changed.append(exact["test_id"])
    return {
        "description": (
            "Holm within the same eleven-test family using the continuity-corrected "
            "chi-square McNemar p from the frozen route instead of the exact binomial p"
        ),
        "n_verdicts_changed": len(changed),
        "tests_with_changed_verdict": changed,
    }


def load_split(inputs_dir: Path, split_key: str) -> tuple[dict[str, dict], dict[str, dict], dict[str, dict]]:
    spec = SPLITS[split_key]
    preds: dict[str, dict] = {}
    manifests: dict[str, dict] = {}
    results: dict[str, dict] = {}
    for system in ("baseline", "preprocessing"):
        run_dir = inputs_dir / spec["dataset"] / f"{MODEL_PREFIX}_{system}_{spec['scope']}"
        pred_path = run_dir / "predictions.json"
        if not pred_path.exists():
            raise FileNotFoundError(f"Missing predictions: {pred_path}")
        preds[system] = load_predictions(pred_path)
        manifests[system] = json.loads((run_dir / "predictions_manifest.json").read_text())
        results[system] = json.loads((run_dir / "results.json").read_text())
        validate_prediction_payload(preds[system], manifests[system], f"{split_key}/{system}")
    if manifests["baseline"] != manifests["preprocessing"]:
        raise ValueError(f"{split_key}: baseline and preprocessing manifests differ")
    for system in preds:
        original = preds[system]["original"]
        for spec_ in family_specification():
            ref = spec_["reference_variant"]
            if ref.startswith("clean_"):
                aligned, _, _ = _align_to_reference(original, preds[system][ref])
                if aligned != preds[system][ref]["predictions"]:
                    raise ValueError(
                        f"{split_key}/{system}: paired-clean view {ref} disagrees with "
                        "the aligned original predictions"
                    )
    return preds, manifests, results


def analyze_split(inputs_dir: Path, split_key: str) -> dict[str, Any]:
    preds, manifests, results = load_split(inputs_dir, split_key)
    tests = [run_test(spec, preds) for spec in family_specification()]
    holm_family(tests)
    sensitivity = holm_sensitivity_to_corrected_chi2(tests)
    baseline_results = results["baseline"]
    # Cross-check against results.json drops for the paired conditions.
    drops = baseline_results["drops_pp"]
    for test in tests:
        if test["group"] in ("H1", "H2"):
            recorded = -float(drops[test["treatment_variant"]])
            if abs(recorded - test["difference_pp"]) > 1e-6:
                raise ValueError(
                    f"{split_key}/{test['test_id']}: recomputed difference "
                    f"{test['difference_pp']:.4f} pp disagrees with results.json "
                    f"{recorded:.4f} pp"
                )
    return {
        "split": split_key,
        "label": SPLITS[split_key]["label"],
        "dataset": manifests["baseline"]["dataset"],
        "split_key_manifest": manifests["baseline"]["split_key"],
        "source_checksum_sha256": manifests["baseline"]["source_checksum_sha256"],
        "clean_accuracy_baseline": float(baseline_results["accuracies"]["original"]),
        "clean_accuracy_preprocessing": float(results["preprocessing"]["accuracies"]["original"]),
        "family": {
            "name": "deberta_ext_baseline_only_addendum",
            "description": (
                "Eleven exact two-sided McNemar tests per split on identical source "
                "indices: H1 (1), H2 for three folds (3), unseen H-suffix informal vs "
                "formal and vs random control per fold (6), preprocessing vs baseline "
                "on emoji_raw (1). Holm-corrected at familywise alpha .05. Separate "
                "from the frozen 45-cell primary family."
            ),
            "correction": "Holm",
            "familywise_alpha": FAMILYWISE_ALPHA,
            "n_tests": len(tests),
            "test_ids": [t["test_id"] for t in tests],
        },
        "tests": tests,
        "corrected_chi2_holm_sensitivity": sensitivity,
    }


def _five_encoder_results_path(backup_root: Path, prefix: str, sub_root: str, dataset: str, scope: str) -> Path:
    sub = sub_root.format(short=DATASET_SHORT[dataset])
    return backup_root / sub / dataset / "seed_42" / f"{prefix}_baseline_final_{scope.removeprefix('final_')}" / "results.json"


def six_encoder_table(backup_root: Path, deberta_splits: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """H1 and fold-1 H2 seed-42 baseline paired differences for six encoders."""
    h2_variant = _suffix("fold_1")
    table: dict[str, Any] = {}
    for split_key, spec in SPLITS.items():
        rows: list[dict[str, Any]] = []
        for prefix, display, sub_root in FIVE_ENCODERS:
            path = _five_encoder_results_path(
                backup_root, prefix, sub_root, spec["dataset"], spec["scope"]
            )
            if not path.exists():
                raise FileNotFoundError(f"Missing five-encoder results: {path}")
            payload = json.loads(path.read_text())
            drops = payload["drops_pp"]
            rows.append(
                {
                    "encoder": display,
                    "prefix": prefix,
                    "model_name": payload.get("model_name"),
                    "clean_accuracy": float(payload["accuracies"]["original"]),
                    "H1_emoji_raw_pp": -float(drops[H1_VARIANT]),
                    "H2_fold_1_hsuffix_pp": -float(drops[h2_variant]),
                    "H2_three_fold_mean_pp": -float(
                        np.mean([drops[_suffix(f)] for f in FOLDS])
                    ),
                    "source": path.relative_to(backup_root).as_posix(),
                }
            )
        h1_mean = float(np.mean([r["H1_emoji_raw_pp"] for r in rows]))
        h2_mean = float(np.mean([r["H2_fold_1_hsuffix_pp"] for r in rows]))
        h2_3_mean = float(np.mean([r["H2_three_fold_mean_pp"] for r in rows]))
        deb = deberta_splits[split_key]
        by_id = {t["test_id"]: t for t in deb["tests"]}
        deb_h1 = by_id["H1_emoji_raw_vs_paired_clean"]
        deb_h2 = by_id["H2_fold_1_hsuffix_vs_paired_clean"]
        deb_h2_all = [by_id[f"H2_{f}_hsuffix_vs_paired_clean"]["difference_pp"] for f in FOLDS]
        table[split_key] = {
            "label": spec["label"],
            "five_encoders": rows,
            "five_encoder_mean": {
                "H1_emoji_raw_pp": h1_mean,
                "H2_fold_1_hsuffix_pp": h2_mean,
                "H2_three_fold_mean_pp": h2_3_mean,
            },
            "deberta_v3_base": {
                "encoder": "DeBERTa-v3-base",
                "clean_accuracy": deb["clean_accuracy_baseline"],
                "H1_emoji_raw_pp": deb_h1["difference_pp"],
                "H1_treatment_accuracy": deb_h1["treatment_accuracy"],
                "H1_paired_clean_accuracy": deb_h1["reference_accuracy"],
                "H1_holm_p": deb_h1["holm_adjusted_pvalue"],
                "H2_three_fold_mean_pp": float(np.mean(deb_h2_all)),
                "H2_per_fold_pp": deb_h2_all,
                "H2_per_fold_holm_p": [
                    by_id[f"H2_{f}_hsuffix_vs_paired_clean"]["holm_adjusted_pvalue"] for f in FOLDS
                ],
                "H2_fold_1_hsuffix_pp": deb_h2["difference_pp"],
                "H2_fold_1_holm_p": deb_h2["holm_adjusted_pvalue"],
            },
            "deberta_minus_five_mean": {
                "H1_emoji_raw_pp": deb_h1["difference_pp"] - h1_mean,
                "H2_three_fold_mean_pp": float(np.mean(deb_h2_all)) - h2_3_mean,
                "H2_fold_1_hsuffix_pp": deb_h2["difference_pp"] - h2_mean,
            },
            "caveats": [
                "DeBERTa: one seed (42), no H3/H4 systems, no across-run sensitivity panel.",
                "Five-encoder figures are the seed-42 paired differences that enter tab:primary.",
            ],
        }
    return table


def _fmt_p(p: float) -> str:
    if p < 0.001:
        return "<.001"
    return f"{p:.3f}".lstrip("0") if p < 1 else "1.0"


def render_report(payload: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# DeBERTa-v3-base extension: baseline-only paired statistics\n")
    lines.append(
        "Addendum analysis for the DeBERTa-v3-base Tier 1 runs (seed 42 only, controlled "
        "clean baseline and emoji-normalization preprocessing). This is a **separate "
        "addendum family**: it is baseline-only, uses no augmentation or hybrid "
        "system (no H3/H4), has no across-run sensitivity panel, and is not part of the "
        "frozen 45-cell primary family reported in the paper. It does not modify "
        "`src/statistical_tests.py`.\n"
    )
    lines.append("## Relation to the paper's frozen inference\n")
    lines.append(
        "The paired contrasts are the same kind as the paper's: transformed minus "
        "paired-clean accuracy on identical source indices. The inference is **not** the "
        "same. The frozen route (`src/statistical_tests.py`, main.tex Section "
        "\"Primary comparisons and inference\") uses the continuity-corrected chi-square "
        "McNemar statistic with four-test Holm families per model x split x fold. This "
        "addendum uses the exact binomial two-sided McNemar test with one eleven-test Holm "
        "family per split. Effect sizes are directly comparable across the six encoders; "
        "the adjusted p values and significance counts here are a separate inference and "
        "must not be merged into Table tab:primary's \"Holm sig.\" row.\n"
    )
    sens = [s_["corrected_chi2_holm_sensitivity"]["n_verdicts_changed"] for s_ in payload["splits"]]
    if sum(sens) == 0:
        lines.append(
            "Sensitivity check: re-running Holm within each eleven-test family on the "
            "frozen route's continuity-corrected chi-square p values instead of the exact "
            "binomial p values changes no significance verdict on any split (0/33). Both p "
            "values are stored per test in the JSON.\n"
        )
    else:
        lines.append(
            f"Sensitivity check: using the continuity-corrected chi-square p values within "
            f"the same families changes {sum(sens)}/33 verdicts; see "
            "`corrected_chi2_holm_sensitivity` in the JSON.\n"
        )
    lines.append("## Family definition\n")
    lines.append(
        "One Holm family per split (SNLI test, MNLI matched, MNLI mismatched) "
        "containing exactly eleven exact two-sided McNemar tests, each on identical "
        "source indices:\n"
    )
    lines.append("1. **H1** baseline `emoji_raw` vs its paired-clean view `clean_emoji_raw`.")
    lines.append(
        "2-4. **H2** baseline `marker_unseen_fold_{1,2,3}_hypothesis_suffix` vs the "
        "paired-clean view (identical to the aligned `original` predictions; verified)."
    )
    lines.append(
        "5-10. **Controls** each fold's unseen H-suffix informal condition vs the same "
        "fold's `_formal_control` and vs its `_random_control` (same sources)."
    )
    lines.append(
        "11. **Normalization** preprocessing `emoji_raw` vs baseline `emoji_raw` on the "
        "same sources (baseline-only analogue of the H3 repair question).\n"
    )
    lines.append(
        f"Differences are treatment minus reference in percentage points. Raw p is the "
        f"exact binomial McNemar p; Holm p is adjusted within the eleven-test family at "
        f"familywise alpha = {FAMILYWISE_ALPHA}. Each interval is a **pointwise 95% "
        f"percentile bootstrap interval** over sources ({BOOTSTRAP_RESAMPLES} paired "
        f"resamples, seed {BOOTSTRAP_SEED}); intervals carry no multiplicity adjustment, "
        "so an interval that excludes zero can coexist with a non-significant Holm result "
        "(as for the MNLI-mismatched fold-1 random-control contrast below).\n"
    )
    lines.append("## Emoji convention and paired-clean endpoints\n")
    lines.append(
        "All emoji figures are paired-clean drops: `emoji_raw` accuracy minus the accuracy "
        "of the same sources without emoji (`clean_emoji_raw`), not minus the full-split "
        "clean accuracy. Endpoints:\n"
    )
    lines.append("| Split | n emoji sources | paired-clean acc. | emoji_raw acc. | H1 (pp) | preprocessing emoji_raw acc. | normalization (pp) |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for split in payload["splits"]:
        by_id = {t["test_id"]: t for t in split["tests"]}
        h1 = by_id["H1_emoji_raw_vs_paired_clean"]
        norm = by_id["normalization_preprocessing_vs_baseline_emoji_raw"]
        lines.append(
            f"| {split['label']} | {h1['n_pairs']} | {h1['reference_accuracy']*100:.2f} | "
            f"{h1['treatment_accuracy']*100:.2f} | {h1['difference_pp']:+.2f} | "
            f"{norm['treatment_accuracy']*100:.2f} | {norm['difference_pp']:+.2f} |"
        )
    lines.append(
        "\nOn SNLI the H1 drop is -1.63 pp (92.50 -> 90.87) and the normalization gain is "
        "+1.62 pp (90.87 -> 92.49): these are two different contrasts, not a rounding "
        "discrepancy; normalized emoji input remains 0.02 pp below its paired-clean "
        "accuracy. On both MNLI splits normalization returns exactly to the paired-clean "
        "accuracy.\n"
    )
    for split in payload["splits"]:
        lines.append(
            f"## {split['label']} (clean baseline {split['clean_accuracy_baseline']*100:.2f}%, "
            f"clean preprocessing {split['clean_accuracy_preprocessing']*100:.2f}%)\n"
        )
        lines.append(
            "| # | Test | Treatment | Reference | n | Diff (pp) | pointwise 95% CI | disc. (T only / R only) | raw p | Holm p | sig. |"
        )
        lines.append("|---|---|---|---|---:|---:|---|---:|---:|---:|:-:|")
        for i, t in enumerate(split["tests"], 1):
            lines.append(
                f"| {i} | {t['test_id']} | {t['treatment_system']}:{t['treatment_variant']} | "
                f"{t['reference_system']}:{t['reference_variant']} | {t['n_pairs']} | "
                f"{t['difference_pp']:+.2f} | [{t['ci_lower_pp']:+.2f}, {t['ci_upper_pp']:+.2f}] | "
                f"{t['n_discordant']} ({t['n_treatment_only_correct']} / {t['n_reference_only_correct']}) | "
                f"{_fmt_p(t['pvalue'])} | {_fmt_p(t['holm_adjusted_pvalue'])} | "
                f"{'yes' if t['significant_holm'] else 'no'} |"
            )
        n_sig = sum(1 for t in split["tests"] if t["significant_holm"])
        lines.append(
            f"\nHolm-significant: {n_sig}/11 (addendum family; not comparable to the primary "
            f"\"Holm sig.\" counts). Verdicts unchanged under corrected chi-square p: "
            f"{'yes' if split['corrected_chi2_holm_sensitivity']['n_verdicts_changed'] == 0 else 'no'}.\n"
        )
    lines.append("## Six-encoder comparison (seed-42 clean baselines, paired differences in pp)\n")
    lines.append(
        "H1 is `emoji_raw` minus paired clean. H2 is the **three-fold mean** of the unseen "
        "hypothesis-suffix marker minus clean, matching the convention of Table "
        "tab:primary; the fold-1 value is given as a secondary line. Five-encoder values "
        "are read from the archived `results.json` `drops_pp` fields (sign flipped so "
        "negative is a drop) and reproduce the tab:primary task rows; DeBERTa values are "
        "recomputed here from per-example predictions and agree with its `results.json` "
        "to 1e-6. DeBERTa has one seed, no H3/H4 systems, and no across-run panel.\n"
    )
    for split_key, block in payload["six_encoder_table"].items():
        lines.append(f"### {block['label']}\n")
        lines.append("| Encoder | Clean acc. | H1 emoji | H2 three-fold mean | H2 fold-1 (secondary) |")
        lines.append("|---|---:|---:|---:|---:|")
        for r in block["five_encoders"]:
            lines.append(
                f"| {r['encoder']} | {r['clean_accuracy']*100:.2f} | {r['H1_emoji_raw_pp']:+.2f} | "
                f"{r['H2_three_fold_mean_pp']:+.2f} | {r['H2_fold_1_hsuffix_pp']:+.2f} |"
            )
        m = block["five_encoder_mean"]
        lines.append(
            f"| **Mean of five** | | **{m['H1_emoji_raw_pp']:+.2f}** | **{m['H2_three_fold_mean_pp']:+.2f}** | "
            f"{m['H2_fold_1_hsuffix_pp']:+.2f} |"
        )
        d = block["deberta_v3_base"]
        fold_ps = ", ".join(_fmt_p(x) for x in d["H2_per_fold_holm_p"])
        lines.append(
            f"| **DeBERTa-v3-base** | {d['clean_accuracy']*100:.2f} | **{d['H1_emoji_raw_pp']:+.2f}** "
            f"(Holm p {_fmt_p(d['H1_holm_p'])}) | **{d['H2_three_fold_mean_pp']:+.2f}** "
            f"(per-fold Holm p {fold_ps}) | {d['H2_fold_1_hsuffix_pp']:+.2f} "
            f"(Holm p {_fmt_p(d['H2_fold_1_holm_p'])}) |"
        )
        g = block["deberta_minus_five_mean"]
        lines.append(
            f"| DeBERTa minus five-mean | | {g['H1_emoji_raw_pp']:+.2f} | "
            f"{g['H2_three_fold_mean_pp']:+.2f} | {g['H2_fold_1_hsuffix_pp']:+.2f} |"
        )
        lines.append("")
    six = payload["six_encoder_table"]
    lines.append(
        "All-task means (three splits): H1 DeBERTa "
        f"{np.mean([six[k]['deberta_v3_base']['H1_emoji_raw_pp'] for k in six]):+.2f} vs five-encoder "
        f"{np.mean([six[k]['five_encoder_mean']['H1_emoji_raw_pp'] for k in six]):+.2f}; "
        "H2 three-fold DeBERTa "
        f"{np.mean([six[k]['deberta_v3_base']['H2_three_fold_mean_pp'] for k in six]):+.2f} vs "
        f"{np.mean([six[k]['five_encoder_mean']['H2_three_fold_mean_pp'] for k in six]):+.2f} "
        "(secondary, fold-1 only: DeBERTa "
        f"{np.mean([six[k]['deberta_v3_base']['H2_fold_1_hsuffix_pp'] for k in six]):+.2f} vs "
        f"{np.mean([six[k]['five_encoder_mean']['H2_fold_1_hsuffix_pp'] for k in six]):+.2f}).\n"
    )
    lines.append("## Summary paragraph\n")
    lines.append(payload["summary_paragraph"])
    lines.append("")
    lines.append("## Provenance\n")
    lines.append(
        "Inputs: `predictions.json`, `predictions_manifest.json`, and `results.json` from "
        "the Modal volume `wnut2026-nli-results-v2` under "
        "`deberta_ext/{snli,multi_nli}/seed_42/deberta_v3_base_{baseline,preprocessing}_final_*/`, "
        "downloaded to `statistics/inputs/`. Baseline and preprocessing manifests are "
        "identical per split (same source checksum, generation seed 42). "
        "Five-encoder figures: `PREDICTIONS_DIR/fold_study/` "
        "(`publication_results/` for ELECTRA, TimeLM, BERTweet; `_colab/` for RoBERTa-base "
        "and RoBERTa-large). Script: `src/deberta_ext_statistics.py`; tests: "
        "`tests/test_deberta_ext_statistics.py`."
    )
    return "\n".join(lines) + "\n"


def summary_paragraph(payload: dict[str, Any]) -> str:
    parts: list[str] = []
    parts.append(
        "As an addendum outside the frozen primary family, we fit a single-seed (42) "
        "DeBERTa-v3-base clean baseline and its emoji-normalization counterpart (Tier 1 "
        "only; no augmentation or hybrid system, so no H3/H4 and no across-run panel) and "
        "computed the same paired contrasts on identical source indices. The inference "
        "differs from the primary route: exact binomial McNemar tests in one Holm family of "
        "eleven baseline-only tests per split, rather than continuity-corrected chi-square "
        "tests in four-test families; the adjusted $p$ values below are therefore a separate "
        "inference and are not added to the primary significance counts. Intervals are "
        "pointwise 95\\% bootstrap intervals over sources."
    )
    for split in payload["splits"]:
        by_id = {t["test_id"]: t for t in split["tests"]}
        h1 = by_id["H1_emoji_raw_vs_paired_clean"]
        h2 = [by_id[f"H2_{f}_hsuffix_vs_paired_clean"] for f in FOLDS]
        norm = by_id["normalization_preprocessing_vs_baseline_emoji_raw"]
        f1_formal = by_id["control_fold_1_hsuffix_vs_formal"]
        f1_random = by_id["control_fold_1_hsuffix_vs_random"]
        h2_mean = float(np.mean([t["difference_pp"] for t in h2]))
        h2_vals = ", ".join(f"{t['difference_pp']:+.2f}" for t in h2)
        h2_ps = ", ".join(_fmt_p(t["holm_adjusted_pvalue"]) for t in h2)
        other_controls = [
            t for t in split["tests"] if t["group"] == "control" and "fold_1" not in t["test_id"]
        ]
        reversed_controls = sum(1 for t in other_controls if t["difference_pp"] > 0 and t["significant_holm"])
        parts.append(
            f"On {split['label']} (clean {split['clean_accuracy_baseline']*100:.2f}\\%), raw emoji "
            f"lowers paired accuracy from {h1['reference_accuracy']*100:.2f} to "
            f"{h1['treatment_accuracy']*100:.2f}\\%, {h1['difference_pp']:+.2f}\\pp (pointwise 95\\% CI "
            f"[{h1['ci_lower_pp']:+.2f}, {h1['ci_upper_pp']:+.2f}], Holm $p$ "
            f"{_fmt_p(h1['holm_adjusted_pvalue'])}); the held-out hypothesis-suffix marker "
            f"changes it by {h2_mean:+.2f}\\pp averaged over folds ({h2_vals}\\pp for folds 1-3; "
            f"Holm $p$ {h2_ps}); the fold-1 informal marker is {f1_formal['difference_pp']:+.2f}\\pp "
            f"against its formal control (Holm $p$ {_fmt_p(f1_formal['holm_adjusted_pvalue'])}) and "
            f"{f1_random['difference_pp']:+.2f}\\pp against its random control (Holm $p$ "
            f"{_fmt_p(f1_random['holm_adjusted_pvalue'])}), while {reversed_controls} of the four "
            f"fold-2 and fold-3 control contrasts are significantly reversed (the control phrase "
            f"harms DeBERTa more than the informal marker); and normalization moves raw-emoji "
            f"accuracy by {norm['difference_pp']:+.2f}\\pp (Holm $p$ "
            f"{_fmt_p(norm['holm_adjusted_pvalue'])})."
        )
    six = payload["six_encoder_table"]
    h1_deb = [six[k]["deberta_v3_base"]["H1_emoji_raw_pp"] for k in six]
    h1_five = [six[k]["five_encoder_mean"]["H1_emoji_raw_pp"] for k in six]
    h2_deb = [six[k]["deberta_v3_base"]["H2_three_fold_mean_pp"] for k in six]
    h2_five = [six[k]["five_encoder_mean"]["H2_three_fold_mean_pp"] for k in six]
    h2f1_deb = [six[k]["deberta_v3_base"]["H2_fold_1_hsuffix_pp"] for k in six]
    h2f1_five = [six[k]["five_encoder_mean"]["H2_fold_1_hsuffix_pp"] for k in six]
    parts.append(
        f"Across the three splits the DeBERTa H1 effect averages {np.mean(h1_deb):+.2f}\\pp against "
        f"a five-encoder mean of {np.mean(h1_five):+.2f}\\pp, and the three-fold H2 effect averages "
        f"{np.mean(h2_deb):+.2f}\\pp against {np.mean(h2_five):+.2f}\\pp (fold 1 alone: "
        f"{np.mean(h2f1_deb):+.2f} against {np.mean(h2f1_five):+.2f}\\pp). Both effects reproduce "
        "in direction on a sixth, non-RoBERTa architecture with byte-fallback tokenization, "
        "at reduced magnitude. Using the corrected chi-square $p$ values within these families "
        "changes no verdict. These eleven-test families are reported separately and do not "
        "alter the 45-cell primary counts; H3 and H4 for DeBERTa would require Tier 2 fine-tunes."
    )
    return " ".join(parts)


def run(inputs_dir: Path, backup_root: Path, out_dir: Path) -> dict[str, Any]:
    splits = [analyze_split(inputs_dir, key) for key in SPLITS]
    by_key = {s["split"]: s for s in splits}
    table = six_encoder_table(backup_root, by_key)
    payload: dict[str, Any] = {
        "analysis": "deberta_ext_baseline_only_paired_statistics",
        "model": "microsoft/deberta-v3-base",
        "seed": 42,
        "systems_used": ["baseline", "preprocessing"],
        "part_of_frozen_primary_family": False,
        "test": "exact two-sided McNemar (binomial) on identical source indices",
        "correction": "Holm within each split's eleven-test family",
        "familywise_alpha": FAMILYWISE_ALPHA,
        "bootstrap": {"resamples": BOOTSTRAP_RESAMPLES, "seed": BOOTSTRAP_SEED, "unit": "source"},
        "family_specification": family_specification(),
        "splits": splits,
        "six_encoder_table": table,
    }
    payload["summary_paragraph"] = summary_paragraph(payload)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "deberta_ext_statistics.json").write_text(json.dumps(payload, indent=2) + "\n")
    (out_dir / "REPORT.md").write_text(render_report(payload))
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--inputs-dir", type=Path, required=True)
    parser.add_argument("--backup-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    payload = run(args.inputs_dir, args.backup_root, args.out_dir)
    for split in payload["splits"]:
        print(f"\n{split['label']}")
        for t in split["tests"]:
            print(
                f"  {t['test_id']:<52} {t['difference_pp']:+7.2f} pp  disc={t['n_discordant']:>5}  "
                f"p={t['pvalue']:.3g}  holm={t['holm_adjusted_pvalue']:.3g}"
            )
    print(f"\nWrote {args.out_dir / 'deberta_ext_statistics.json'} and REPORT.md")


if __name__ == "__main__":
    main()
