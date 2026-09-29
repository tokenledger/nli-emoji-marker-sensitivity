"""
Statistical significance testing for all approach comparisons.
Requires predictions.json files from each training script.

Runs:
  - McNemar's test (paired, per variant) between every approach pair
  - Bootstrap 95% confidence intervals on every accuracy number

Usage:
    python src/statistical_tests.py --results_dir ./results --model_prefix electra
    python src/statistical_tests.py --results_dir ./results --model_prefix roberta
"""
import argparse
import json
from pathlib import Path
from itertools import combinations

import numpy as np
from statsmodels.stats.contingency_tables import mcnemar

from transforms import PSYCH_LABEL_POLICY


def load_predictions(path):
    with open(path) as f:
        return json.load(f)


def validate_prediction_payload(payload, identity, name):
    expected = {'original', *identity.get('evaluated_variants', [])}
    observed = {variant for variant in payload if not variant.startswith('clean_')}
    if observed != expected:
        raise ValueError(
            f'{name} prediction variants {sorted(observed)} do not match manifest '
            f'{sorted(expected)}'
        )
    for variant in expected:
        record = payload[variant]
        predictions = record.get('predictions', [])
        labels = record.get('labels', [])
        source_ids = record.get('source_indices')
        if source_ids is None:
            raise ValueError(f'{name}/{variant} lacks source_indices')
        if not (len(predictions) == len(labels) == len(source_ids)):
            raise ValueError(f'{name}/{variant} has unequal prediction arrays')
        if not predictions:
            raise ValueError(f'{name}/{variant} has no predictions')
        if len(source_ids) != len(set(source_ids)):
            raise ValueError(f'{name}/{variant} has duplicate source_indices')


def mcnemar_test(preds_a, preds_b, labels):
    """McNemar's test: do models A and B make significantly different errors?"""
    a = np.array(preds_a) == np.array(labels)
    b = np.array(preds_b) == np.array(labels)
    n_both     = int((a & b).sum())
    n_a_only   = int((a & ~b).sum())
    n_b_only   = int((~a & b).sum())
    n_neither  = int((~a & ~b).sum())
    table = [[n_both, n_a_only], [n_b_only, n_neither]]
    if n_a_only + n_b_only == 0:
        statistic, pvalue = 0.0, 1.0
    else:
        fitted = mcnemar(table, exact=False, correction=True)
        statistic, pvalue = float(fitted.statistic), float(fitted.pvalue)
    return {
        'statistic': statistic,
        'pvalue':    pvalue,
        'significant_05': bool(pvalue < 0.05),
        'n_a_wins': n_a_only,
        'n_b_wins': n_b_only,
    }


def bootstrap_ci(predictions, labels, n_bootstrap=2000, ci=0.95):
    """Bootstrap confidence interval for accuracy."""
    preds  = np.array(predictions)
    labels = np.array(labels)
    n      = len(preds)
    accs   = []
    rng    = np.random.default_rng(seed=42)
    for _ in range(n_bootstrap):
        idx  = rng.integers(0, n, n)
        accs.append((preds[idx] == labels[idx]).mean())
    lo = float(np.percentile(accs, (1 - ci) / 2 * 100))
    hi = float(np.percentile(accs, (1 + ci) / 2 * 100))
    point_estimate = float(np.mean(preds == labels))
    return {
        'point_estimate': point_estimate,
        # Retain ``mean`` for report compatibility, but make it the actual
        # sample estimate rather than the Monte Carlo mean of bootstrap draws.
        'mean': point_estimate,
        'bootstrap_mean': float(np.mean(accs)),
        'lower':  lo,
        'upper':  hi,
    }


def _holm_adjust(entries, alpha=0.05):
    ordered = sorted(entries, key=lambda item: item[2]['pvalue'])
    running = 0.0
    total = len(ordered)
    for index, (_, _, result) in enumerate(ordered):
        adjusted = min(1.0, (total - index) * result['pvalue'])
        running = max(running, adjusted)
        result['holm_adjusted_pvalue'] = running
        result['significant_holm'] = bool(running < alpha)


def _align_to_reference(candidate, reference):
    """Align a full prediction record to a paired transformed subset."""

    candidate_ids = candidate['source_indices']
    reference_ids = reference['source_indices']
    positions = {source_id: index for index, source_id in enumerate(candidate_ids)}
    missing = [source_id for source_id in reference_ids if source_id not in positions]
    if missing:
        raise ValueError(f'Paired comparison is missing source IDs: {missing[:5]}')
    selected = [positions[source_id] for source_id in reference_ids]
    candidate_labels = [candidate['labels'][index] for index in selected]
    if candidate_labels != reference['labels']:
        raise ValueError('Paired comparison labels differ after source-ID alignment')
    return (
        [candidate['predictions'][index] for index in selected],
        reference['predictions'],
        reference['labels'],
    )


def _resolve_primary_variant(variants, candidates, description):
    """Resolve a frozen condition across canonical and compatibility names."""

    for candidate in candidates:
        if candidate in variants:
            return candidate
    raise ValueError(
        f'Frozen primary family requires {description}; expected one of '
        f'{list(candidates)}'
    )


def _variant_label_policy(identity, variant):
    return identity.get('variant_checksums', {}).get(variant, {}).get(
        'label_policy', 'preserve'
    )


def run_tests(results_dir, model_prefix, marker_fold='fold_1', eval_role='final',
              eval_split='test', output_path=None):
    approaches = [
        'baseline', 'augmented', 'clean_control',
        'preprocessing', 'marker_oracle', 'hybrid',
    ]

    # Load per-example predictions for each approach
    preds = {}
    identities = {}
    directory_names = {
        'baseline': 'baseline',
        'augmented': f'augmented_{marker_fold}',
        'clean_control': 'clean_control',
        'preprocessing': 'preprocessing',
        'marker_oracle': 'marker_oracle',
        'hybrid': f'hybrid_{marker_fold}',
    }
    scope = f'{eval_role}_{str(eval_split).replace("/", "_").replace(":", "_")}'
    for approach in approaches:
        path = Path(results_dir) / f'{model_prefix}_{directory_names[approach]}_{scope}' / 'predictions.json'
        if not path.exists():
            raise FileNotFoundError(f'Missing required production-system predictions: {path}')
        preds[approach] = load_predictions(path)
        manifest_path = path.with_name('predictions_manifest.json')
        if not manifest_path.exists():
            raise FileNotFoundError(f'Missing prediction identity: {manifest_path}')
        identities[approach] = json.loads(manifest_path.read_text(encoding='utf-8'))
        validate_prediction_payload(preds[approach], identities[approach], approach)

    available = list(preds.keys())
    print(f'Loaded predictions for: {available}')
    if not available:
        raise FileNotFoundError('No approach prediction files were found')
    first_identity = identities[available[0]]
    for approach in available[1:]:
        if identities[approach] != first_identity:
            raise ValueError(
                f'Prediction manifest mismatch: {available[0]} versus {approach}'
            )
    evaluated_variants = list(first_identity['evaluated_variants'])
    psych_variants = [
        variant for variant in evaluated_variants
        if _variant_label_policy(first_identity, variant) == PSYCH_LABEL_POLICY
    ]
    variants = [
        'original',
        *[variant for variant in evaluated_variants if variant not in psych_variants],
    ]
    emoji_variant = _resolve_primary_variant(
        variants, ('emoji_raw', 'emoji'), 'the bijective emoji condition'
    )
    marker_variant = _resolve_primary_variant(
        variants,
        (f'marker_unseen_{marker_fold}_hypothesis_suffix',),
        f'the held-out marker condition for {marker_fold}',
    )
    combined_candidates = [f'emoji_marker_combined_{marker_fold}']
    if marker_fold == 'fold_1':
        combined_candidates.insert(0, 'emoji_marker_combined')
    combined_variant = _resolve_primary_variant(
        variants, combined_candidates, f'the combined condition for {marker_fold}'
    )
    print('Primary family: 4 preregistered paired tests; Holm familywise alpha=0.05')

    all_stats = {}

    # --- Confidence intervals for every (approach, variant) ---
    print('\n=== Bootstrap 95% Confidence Intervals ===')
    print(f'{"":18}' + ''.join(f'{v:>24}' for v in variants))
    print('-' * (18 + 24 * len(variants)))

    ci_results = {}
    for approach in available:
        row = f'{approach:<18}'
        ci_results[approach] = {}
        for variant in variants:
            if variant not in preds[approach]:
                row += f'{"N/A":>24}'
                continue
            p = preds[approach][variant]
            ci = bootstrap_ci(p['predictions'], p['labels'])
            ci_results[approach][variant] = ci
            interval = (
                f'{ci["point_estimate"] * 100:.2f} '
                f'[{ci["lower"] * 100:.2f},{ci["upper"] * 100:.2f}]'
            )
            row += f'{interval:>24}'
        print(row)

    all_stats['confidence_intervals'] = ci_results

    # --- Four frozen primary hypotheses ---
    primary_specs = [
        ('H1_emoji_vs_clean_baseline', 'baseline', 'original', 'baseline', emoji_variant),
        ('H2_marker_vs_clean_baseline', 'baseline', 'original', 'baseline', marker_variant),
        ('H3_emoji_normalization_vs_augmentation', 'preprocessing', emoji_variant,
         'augmented', emoji_variant),
        ('H4_hybrid_vs_baseline_combined', 'hybrid', combined_variant,
         'baseline', combined_variant),
    ]
    primary_results = {}
    primary_entries = []
    for hypothesis, approach_a, variant_a, approach_b, variant_b in primary_specs:
        aligned_a, aligned_b, labels = _align_to_reference(
            preds[approach_a][variant_a], preds[approach_b][variant_b]
        )
        result = mcnemar_test(aligned_a, aligned_b, labels)
        result.update({
            'system_a': approach_a, 'variant_a': variant_a,
            'system_b': approach_b, 'variant_b': variant_b,
        })
        primary_results[hypothesis] = result
        primary_entries.append((hypothesis, '', result))
    _holm_adjust(primary_entries)

    # --- Larger exploratory family: every approach pair on every variant ---
    print("\n=== Exploratory McNemar tests (separate Holm family) ===")
    pairs = list(combinations(available, 2))
    exploratory_results = {}
    for variant in variants:
        print(f'\nVariant: {variant}')
        for a, b in pairs:
            if variant not in preds[a] or variant not in preds[b]:
                continue
            labels_a = preds[a][variant]['labels']
            labels_b = preds[b][variant]['labels']
            ids_a = preds[a][variant].get('source_indices')
            ids_b = preds[b][variant].get('source_indices')
            # Verify same eval set
            if labels_a != labels_b:
                raise ValueError(
                    f'{a} and {b} have different labels for {variant}'
                )
            if ids_a is not None and ids_b is not None and ids_a != ids_b:
                raise ValueError(
                    f'{a} and {b} have different source IDs for {variant}'
                )
            result = mcnemar_test(
                preds[a][variant]['predictions'],
                preds[b][variant]['predictions'],
                labels_a,
            )
            print(f'  {a} vs {b}: raw p={result["pvalue"]:.4f}  '
                  f'(A wins {result["n_a_wins"]}, B wins {result["n_b_wins"]})')
            key = f'{a}_vs_{b}'
            exploratory_results.setdefault(key, {})[variant] = result

    exploratory_entries = [
        (pair, variant, result)
        for pair, by_variant in exploratory_results.items()
        for variant, result in by_variant.items()
    ]
    _holm_adjust(exploratory_entries)

    all_stats['primary'] = {
        'family': 'four preregistered hypotheses',
        'correction': 'Holm', 'familywise_alpha': 0.05,
        'n_tests': len(primary_results), 'tests': primary_results,
    }
    all_stats['exploratory'] = {
        'family': 'all approach pairs by evaluated variant',
        'correction': 'Holm', 'familywise_alpha': 0.05,
        'n_tests': len(exploratory_entries), 'tests': exploratory_results,
    }
    all_stats['psych_instruction_inversion'] = {
        'family': 'label-changing instruction inversion reported separately',
        'gold_label_policy': PSYCH_LABEL_POLICY,
        'included_in_primary_holm_family': False,
        'included_in_exploratory_holm_family': False,
        'conditions': {
            variant: {
                approach: {
                    'accuracy': float(np.mean(
                        np.asarray(preds[approach][variant]['predictions'])
                        == np.asarray(preds[approach][variant]['labels'])
                    )),
                    'confidence_interval': bootstrap_ci(
                        preds[approach][variant]['predictions'],
                        preds[approach][variant]['labels'],
                    ),
                }
                for approach in available
            }
            for variant in psych_variants
        },
    }

    # --- Save ---
    safe_split = str(eval_split).replace('/', '_').replace(':', '_')
    out_path = Path(output_path) if output_path else (
        Path(results_dir)
        / f'{model_prefix}_statistical_tests_{eval_role}_{safe_split}_{marker_fold}.json'
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(all_stats, f, indent=2)
    print(f'\nFull stats saved to {out_path}')

    return all_stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_dir', type=str, default='./results')
    parser.add_argument('--model_prefix', type=str, default='electra',
                        help='Prefix of result folders, e.g. "electra" matches '
                             'electra_baseline, electra_augmented, etc.')
    parser.add_argument('--marker_fold', default='fold_1',
                        choices=['fold_1', 'fold_2', 'fold_3'])
    parser.add_argument('--eval_role', choices=['development', 'final'], default='final')
    parser.add_argument('--eval_split', required=True)
    parser.add_argument('--output_path', default=None,
                        help='Destination file; defaults to a file inside --results_dir.')
    args = parser.parse_args()
    run_tests(
        args.results_dir, args.model_prefix, args.marker_fold,
        args.eval_role, args.eval_split, args.output_path,
    )


if __name__ == '__main__':
    main()
