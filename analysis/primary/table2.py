"""Assemble Table 2 of the paper from the outputs of the primary analysis.

Inputs below OUTPUT_DIR/primary:
  tables/primary_results.csv                 four prespecified contrasts per cell
  tables/emoji_diagnostics.csv               emoji loss and repair by normalization
  tables/augmentation_transfer_diagnostics.csv  gain from augmentation
  statistics/*.json                          clean accuracy on the full splits
  deberta/deberta_ext_statistics.json        DeBERTa-v3-base (separate family)

Every entry is a mean of per-cell values in percentage points: over the three
splits for the emoji columns, and over splits and folds for the other columns.
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

ENCODERS = (
    ('electra', 'ELECTRA-small'),
    ('roberta_base', 'RoBERTa-base'),
    ('roberta', 'RoBERTa-large'),
    ('timelm', 'TimeLM-21'),
    ('bertweet', 'BERTweet'),
)
SPLITS = (
    ('snli', 'test'),
    ('multi_nli', 'validation_matched'),
    ('multi_nli', 'validation_mismatched'),
)
COLUMNS = (
    'encoder', 'clean_snli', 'clean_mnli_matched', 'clean_mnli_mismatched',
    'emoji_loss_pp', 'emoji_repair_pp', 'marker_loss_pp', 'marker_gain_pp',
    'marker_loss_recovered_pct', 'normalization_minus_augmentation_pp',
    'hybrid_minus_baseline_pp',
)


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline='', encoding='utf-8') as handle:
        return list(csv.DictReader(handle))


def five_encoder_rows(root: Path) -> tuple[list[dict], dict[str, int]]:
    primary = read_csv(root / 'tables' / 'primary_results.csv')
    emoji = read_csv(root / 'tables' / 'emoji_diagnostics.csv')
    transfer = read_csv(root / 'tables' / 'augmentation_transfer_diagnostics.csv')
    contrasts: dict[tuple[str, str], list[float]] = defaultdict(list)
    significant: dict[str, set] = defaultdict(set)
    for row in primary:
        hypothesis = row['hypothesis']
        cell = (row['model'], row['dataset'], row['split'])
        if hypothesis != 'H1':
            cell += (row['fold'],)
        elif row['fold'] != 'fold_1':
            continue
        contrasts[(row['model'], hypothesis)].append(float(row['difference_pp']))
        if row['significant_holm'] == 'yes':
            significant[hypothesis].add(cell)
    rows = []
    for model, label in ENCODERS:
        clean = []
        for dataset, split in SPLITS:
            statistics = json.loads(
                (root / 'statistics' / f'{model}_{dataset}_{split}_fold_1.json')
                .read_text(encoding='utf-8')
            )
            interval = statistics['confidence_intervals']['baseline']['original']
            clean.append(100 * interval['point_estimate'])
        repair = [
            100 * (float(row['normalization_accuracy']) - float(row['emoji_raw_accuracy']))
            for row in emoji if row['model'] == model and row['fold'] == 'fold_1'
        ]
        gain = [
            float(row['difference_pp']) for row in transfer
            if row['model'] == model and row['treatment_system'] == 'augmented'
            and row['marker_status'] == 'unseen' and row['placement'] == 'hypothesis_suffix'
        ]
        if len(repair) != 3 or len(gain) != 9:
            raise SystemExit(
                f'Incomplete primary tables for {model}: {len(repair)} emoji cells, '
                f'{len(gain)} augmentation cells'
            )
        loss = mean(contrasts[(model, 'H2')])
        rows.append({
            'encoder': label,
            'clean_snli': clean[0],
            'clean_mnli_matched': clean[1],
            'clean_mnli_mismatched': clean[2],
            'emoji_loss_pp': mean(contrasts[(model, 'H1')]),
            'emoji_repair_pp': mean(repair),
            'marker_loss_pp': loss,
            'marker_gain_pp': mean(gain),
            'marker_loss_recovered_pct': 100 * mean(gain) / -loss,
            'normalization_minus_augmentation_pp': mean(contrasts[(model, 'H3')]),
            'hybrid_minus_baseline_pp': mean(contrasts[(model, 'H4')]),
        })
    return rows, {key: len(value) for key, value in significant.items()}


def mean_row(rows: list[dict]) -> dict:
    summary = {'encoder': 'Mean of five'}
    for column in COLUMNS[4:]:
        summary[column] = mean([row[column] for row in rows])
    loss, gain = summary['marker_loss_pp'], summary['marker_gain_pp']
    summary['marker_loss_recovered_pct'] = 100 * gain / -loss
    return summary


def deberta_row(root: Path) -> dict:
    payload = json.loads(
        (root / 'deberta' / 'deberta_ext_statistics.json').read_text(encoding='utf-8')
    )
    clean, loss, repair, markers = [], [], [], []
    for split in payload['splits']:
        tests = {test['test_id']: test for test in split['tests']}
        clean.append(100 * split['clean_accuracy_baseline'])
        loss.append(tests['H1_emoji_raw_vs_paired_clean']['difference_pp'])
        repair.append(
            tests['normalization_preprocessing_vs_baseline_emoji_raw']['difference_pp']
        )
        markers.extend(
            tests[f'H2_fold_{fold}_hsuffix_vs_paired_clean']['difference_pp']
            for fold in (1, 2, 3)
        )
    return {
        'encoder': 'DeBERTa-v3-base',
        'clean_snli': clean[0],
        'clean_mnli_matched': clean[1],
        'clean_mnli_mismatched': clean[2],
        'emoji_loss_pp': mean(loss),
        'emoji_repair_pp': mean(repair),
        'marker_loss_pp': mean(markers),
    }


def write(root: Path) -> None:
    rows, significant = five_encoder_rows(root)
    table = rows + [mean_row(rows), deberta_row(root)]
    with (root / 'table2.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS, restval='')
        writer.writeheader()
        writer.writerows(table)
    lines = [
        '# Table 2: per-encoder effects at seed 42',
        '',
        '| ' + ' | '.join(COLUMNS) + ' |',
        '|' + '---|' * len(COLUMNS),
    ]
    for row in table:
        cells = [
            row[column] if column == 'encoder'
            else '' if column not in row
            else f'{row[column]:.1f}' if column == 'marker_loss_recovered_pct'
            else f'{row[column]:.2f}' if column.startswith('clean_')
            else f'{row[column]:+.2f}'
            for column in COLUMNS
        ]
        lines.append('| ' + ' | '.join(cells) + ' |')
    lines += [
        '',
        'Holm-significant cells of the five encoders: '
        f"emoji loss {significant.get('H1', 0)}/15, "
        f"marker loss {significant.get('H2', 0)}/45, "
        f"normalization minus augmentation {significant.get('H3', 0)}/45, "
        f"hybrid minus baseline {significant.get('H4', 0)}/45.",
        '',
    ]
    (root / 'table2.md').write_text('\n'.join(lines), encoding='utf-8')
    print('\n'.join(lines))
