"""NLI dataset utilities shared across all training scripts."""

import json
import hashlib
from pathlib import Path

import numpy as np

from transforms import (PSYCH_LABEL_MAP, PSYCH_LABEL_POLICY, PSYCH_MARKER,
                        invert_from_provenance)


EVAL_VARIANTS = ('slang', 'emoji', 'noise', 'combined')
LABEL_POLICY_PRESERVE = 'preserve'
DEFAULT_DATA_SPLIT_SEED = 42
HUGGINGFACE_DATASET_NAMES = {
    "snli": "stanfordnlp/snli",
    "multi_nli": "nyu-mll/multi_nli",
}


def huggingface_dataset_name(dataset_key):
    """Resolve a logical experiment dataset to its canonical Hub repository."""

    try:
        return HUGGINGFACE_DATASET_NAMES[dataset_key]
    except KeyError as error:
        raise ValueError(f"Unknown NLI dataset key: {dataset_key!r}") from error


def smoke_nli_records(size, *, offset=0):
    """Create deterministic local NLI rows for dependency-free smoke routes."""

    templates = (
        (
            "A man holds a camera while a dog waits in the park.",
            "A person holds a camera.",
            0,
        ),
        (
            "A woman rides a bicycle beside a child.",
            "A bird watches the woman.",
            1,
        ),
        (
            "A boy plays guitar near a car.",
            "No person is playing guitar.",
            2,
        ),
    )
    rows = {"premise": [], "hypothesis": [], "label": []}
    for index in range(offset, offset + size):
        premise, hypothesis, label = templates[index % len(templates)]
        rows["premise"].append(f"{premise} Example {index}.")
        rows["hypothesis"].append(hypothesis)
        rows["label"].append(label)
    return rows


def smoke_nli_splits(datasets_module, *, train_size=100, validation_size=100):
    """Return local Arrow datasets without contacting a dataset registry."""

    dataset_type = datasets_module.Dataset
    return (
        dataset_type.from_dict(smoke_nli_records(train_size)),
        dataset_type.from_dict(smoke_nli_records(validation_size, offset=train_size)),
    )


def limit_dataset(dataset, max_samples=None):
    """Select a deterministic prefix for an explicitly limited integration run."""

    if max_samples is None:
        return dataset
    max_samples = int(max_samples)
    if max_samples < 1:
        raise ValueError('max_samples must be positive')
    return dataset.select(range(min(max_samples, len(dataset))))


TRAIN_SIZE_FULL = 'full'


def parse_train_size(value):
    """Parse ``--train_size``: ``full`` or a positive number of training pairs."""

    if value == TRAIN_SIZE_FULL:
        return TRAIN_SIZE_FULL
    try:
        size = int(value)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"train_size must be 'full' or a positive integer, got {value!r}"
        ) from error
    if size < 1:
        raise ValueError(f'train_size must be positive, got {size}')
    return size


def select_training_subset(dataset, train_size, seed):
    """Draw a deterministic label-stratified training subset.

    Returns the selected rows in their original order and a record for the run
    metadata. Each label receives its proportional share of ``train_size``
    (largest remainders first, ties to the smaller label), and the rows of a
    label are drawn without replacement from a generator seeded with ``seed``.
    The subset is drawn from clean training pairs, before any augmentation.
    """

    train_size = parse_train_size(train_size)
    source_examples = len(dataset)
    labels = np.asarray(dataset['label'])
    if train_size == TRAIN_SIZE_FULL:
        values, counts = np.unique(labels, return_counts=True)
        return dataset, {
            'train_size': TRAIN_SIZE_FULL,
            'is_subset': False,
            'source_examples': source_examples,
            'selected_examples': source_examples,
            'label_counts': {
                str(int(value)): int(count) for value, count in zip(values, counts)
            },
        }
    if train_size > source_examples:
        raise ValueError(
            f'train_size {train_size} exceeds the {source_examples} available '
            'training pairs'
        )
    values, counts = np.unique(labels, return_counts=True)
    exact = counts * (train_size / source_examples)
    allocation = np.floor(exact).astype(int)
    remainder_order = sorted(
        range(len(values)), key=lambda position: (-(exact - allocation)[position], position)
    )
    for position in remainder_order[:train_size - int(allocation.sum())]:
        allocation[position] += 1
    generator = np.random.default_rng(int(seed))
    selected = []
    for value, quota in zip(values, allocation):
        positions = np.flatnonzero(labels == value)
        selected.append(generator.choice(positions, size=int(quota), replace=False))
    indices = np.sort(np.concatenate(selected))
    digest = hashlib.sha256(
        ','.join(str(int(index)) for index in indices).encode('ascii')
    ).hexdigest()
    return dataset.select(indices.tolist()), {
        'train_size': train_size,
        'is_subset': True,
        'source_examples': source_examples,
        'selected_examples': int(len(indices)),
        'selection_seed': int(seed),
        'selection': 'label-stratified, proportional allocation, without replacement',
        'label_counts': {
            str(int(value)): int(quota) for value, quota in zip(values, allocation)
        },
        'source_indices_sha256': digest,
    }


def recorded_training_subset(run_dir):
    """Return the training-subset record stored with a trained run, if any."""

    results_path = Path(run_dir) / 'results.json'
    if not results_path.is_file():
        return None
    results = json.loads(results_path.read_text(encoding='utf-8'))
    return results.get('training_subset')


def limit_eval_suite(eval_datasets, max_samples=None):
    """Limit paired conditions and retain the original rows needed to index them."""

    if max_samples is None:
        return eval_datasets
    limited = {
        name: limit_dataset(dataset, max_samples)
        for name, dataset in eval_datasets.items()
        if name != 'original'
    }
    original = eval_datasets.get('original')
    if original is None:
        return limited
    if 'source_index' not in original.column_names:
        # Sparse ``select`` preserves rows but otherwise loses their identity:
        # downstream evaluation would renumber the selected rows 0..N-1 and
        # paired clean controls could no longer find source IDs such as 141.
        original = original.add_column('source_index', list(range(len(original))))
    required_source_ids = {
        int(source_id)
        for name, dataset in limited.items()
        if not name.startswith('clean_') and 'source_index' in dataset.column_names
        for source_id in dataset['source_index']
    }
    if not required_source_ids:
        limited['original'] = limit_dataset(original, max_samples)
    else:
        original_ids = [int(value) for value in original['source_index']]
        positions = [
            position for position, source_id in enumerate(original_ids)
            if source_id in required_source_ids
        ]
        if len(positions) != len(required_source_ids):
            raise ValueError('Limited evaluation conditions reference missing original rows')
        limited['original'] = original.select(positions)
    return {'original': limited.pop('original'), **limited}


def _timelm_preprocess_text(text):
    """Apply the TimeLM model card's username/link placeholder convention."""

    normalized = []
    for token in text.split():
        if len(token) > 1:
            if token[0] == '@' and token.count('@') == 1:
                token = '@user'
            elif token.startswith('http'):
                token = 'http'
        normalized.append(token)
    return ' '.join(normalized)


def model_input_text(text, profile='none'):
    if profile == 'none':
        return text
    if profile == 'timelm':
        return _timelm_preprocess_text(text)
    raise ValueError(f'Unknown model input preprocessing profile: {profile!r}')


def prepare_dataset_nli(examples, tokenizer, max_seq_length=None,
                        input_preprocessing='none'):
    """Tokenize premise/hypothesis pairs for NLI classification."""
    max_seq_length = tokenizer.model_max_length if max_seq_length is None else max_seq_length
    premise = examples['premise']
    hypothesis = examples['hypothesis']
    if isinstance(premise, list):
        premise = [model_input_text(value, input_preprocessing) for value in premise]
        hypothesis = [model_input_text(value, input_preprocessing) for value in hypothesis]
    else:
        premise = model_input_text(premise, input_preprocessing)
        hypothesis = model_input_text(hypothesis, input_preprocessing)
    tokenized = tokenizer(
        premise,
        hypothesis,
        truncation=True,
        max_length=max_seq_length,
        # Dynamic batch padding is substantially faster and avoids storing
        # millions of padding tokens in model-specific caches.
        padding=False,
    )
    tokenized['label'] = examples['label']
    return tokenized


def map_dataset_locally(dataset, function, *, cache_dir=None, operation,
                        batched=False, remove_columns=None, desc=None,
                        num_proc=None, with_indices=False):
    """Run ``Dataset.map`` without ever writing cache files beside frozen data."""

    kwargs = {
        'batched': batched,
        'remove_columns': remove_columns,
        'desc': desc,
        'with_indices': with_indices,
    }
    if num_proc and int(num_proc) > 1:
        kwargs['num_proc'] = int(num_proc)
    if cache_dir:
        cache_root = Path(cache_dir)
        cache_root.mkdir(parents=True, exist_ok=True)
        identity = json.dumps({
            'fingerprint': getattr(dataset, '_fingerprint', None),
            'operation': operation,
        }, sort_keys=True, separators=(',', ':'))
        key = hashlib.sha256(identity.encode('utf-8')).hexdigest()
        kwargs['cache_file_name'] = str(cache_root / f'{key}.arrow')
        kwargs['load_from_cache_file'] = True
    else:
        # A loaded Arrow dataset otherwise writes cache-*.arrow inside its
        # source directory, which previously inflated Drive by ~35 GB.
        kwargs['keep_in_memory'] = True
        kwargs['load_from_cache_file'] = False
    return dataset.map(function, **kwargs)


def compute_accuracy(eval_preds):
    """Compute classification accuracy for transformers Trainer."""
    return {
        'accuracy': (
            np.argmax(eval_preds.predictions, axis=1) == eval_preds.label_ids
        ).astype(np.float32).mean().item()
    }


def paired_clean_key(variant):
    return f'clean_{variant}'


def evaluation_condition_metadata(name, *, label_policy=None):
    """Return stable reporting metadata for one evaluation condition."""

    paired_of = name[6:] if name.startswith('clean_') else None
    condition = paired_of or name
    resolved_policy = label_policy or (
        PSYCH_LABEL_POLICY if condition == 'psych' and paired_of is None
        else LABEL_POLICY_PRESERVE
    )
    if resolved_policy not in {LABEL_POLICY_PRESERVE, PSYCH_LABEL_POLICY}:
        raise ValueError(f'Unknown evaluation label policy: {resolved_policy!r}')
    label_changing = resolved_policy == PSYCH_LABEL_POLICY
    if paired_of:
        family = 'paired_clean_control'
    elif label_changing:
        family = 'psych_instruction_inversion'
    elif name == 'original':
        family = 'clean_source'
    else:
        family = 'label_preserving_robustness'
    return {
        'condition': name,
        'label_policy': resolved_policy,
        'label_changing': label_changing,
        'evaluation_family': family,
        **({'paired_clean_control_of': paired_of} if paired_of else {}),
    }


def attach_condition_metadata(results):
    """Annotate in-memory metric/prediction records before serialization."""

    for name, record in results.items():
        if not isinstance(record, dict):
            continue
        declared = record.get('condition_metadata', {}).get('label_policy')
        record['condition_metadata'] = evaluation_condition_metadata(
            name, label_policy=declared
        )
    return results


def psych_inversion_result(results):
    """Report psych accuracy against its explicitly inverted gold labels."""

    record = results.get('psych')
    if record is None:
        return None
    accuracy = record.get('accuracy')
    if accuracy is None:
        predictions = np.asarray(record['predictions'])
        labels = np.asarray(record['labels'])
        accuracy = float(np.mean(predictions == labels))
    return {
        'condition': 'psych',
        'accuracy': float(accuracy),
        'gold_label_policy': PSYCH_LABEL_POLICY,
        'label_changing': True,
        'evaluation_family': 'psych_instruction_inversion',
        'paired_by': 'source_index',
        'included_in_robustness_drops': False,
        'included_in_primary_holm_family': False,
    }


def paired_clean_result_from_original(original_result, clean_dataset, *, save_logits=False):
    """Slice original predictions for an exact source-index clean control."""

    indices = [int(value) for value in clean_dataset['source_index']]
    original_indices = [int(value) for value in original_result['source_indices']]
    positions = {
        source_index: position
        for position, source_index in enumerate(original_indices)
    }
    if len(positions) != len(original_indices):
        raise ValueError('Original evaluation source indices are not unique')
    try:
        selected_positions = [positions[index] for index in indices]
    except KeyError as error:
        raise ValueError(
            f'Paired clean source index is absent from original: {error}'
        ) from error
    predictions = [
        original_result['predictions'][position] for position in selected_positions
    ]
    labels = [original_result['labels'][position] for position in selected_positions]
    declared_labels = [int(value) for value in clean_dataset['label']]
    if labels != declared_labels:
        raise ValueError('Paired clean labels do not match original evaluation labels')
    result = {
        'accuracy': float(np.mean(np.asarray(predictions) == np.asarray(labels))),
        'predictions': predictions,
        'labels': labels,
        'source_indices': indices,
        'prediction_source': 'indexed_from_original',
    }
    if save_logits and 'logits' in original_result:
        result['logits'] = [
            original_result['logits'][position] for position in selected_positions
        ]
    return result


def paired_clean_result(results, name, clean_dataset, *, save_logits=False):
    """Build a paired clean result, failing if original was not evaluated first."""

    if not name.startswith('clean_'):
        raise ValueError(f'Paired clean result requires a clean_* condition, got {name!r}')
    if 'original' not in results:
        raise ValueError(
            f'{name!r} requires original predictions to be evaluated first; '
            'refusing an order-dependent extra forward pass'
        )
    return paired_clean_result_from_original(
        results['original'], clean_dataset, save_logits=save_logits
    )


def logical_dataset_checksum(rows):
    """Hash logical rows identically across supported Arrow serialization versions."""

    digest = hashlib.sha256()
    for row in rows:
        payload = json.dumps(
            dict(row), sort_keys=True, separators=(',', ':'), ensure_ascii=False,
            default=lambda value: value.item() if hasattr(value, 'item') else str(value),
        )
        digest.update(payload.encode('utf-8'))
        digest.update(b'\n')
    return digest.hexdigest()


def _read_source_manifest(data_release):
    """Read the immutable source-data manifest from a data release."""

    root = Path(data_release)
    manifest_path = root / 'source_manifest.json'
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f'Frozen source manifest not found: {manifest_path}. '
            'Build it with src/prepare_data_release.py.'
        )
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    if manifest.get('schema_version') != 1:
        raise ValueError(
            f'Unsupported source manifest schema: {manifest.get("schema_version")!r}'
        )
    return root, manifest_path, manifest, hashlib.sha256(manifest_bytes).hexdigest()


def source_dataset_identity(data_release, dataset):
    """Return the model-independent source identity recorded for a run."""

    _, _, manifest, manifest_sha = _read_source_manifest(data_release)
    try:
        entry = manifest['datasets'][dataset]
    except KeyError as error:
        raise ValueError(f'Dataset {dataset!r} is absent from the data release') from error
    return {
        'source_manifest_sha256': manifest_sha,
        'source_config_sha256': manifest['config_sha256'],
        'data_contract_sha256': manifest.get('data_contract_sha256'),
        'huggingface_repository': entry['huggingface_repository'],
        'huggingface_revision': entry['huggingface_revision'],
        'source_exclusions': entry.get('source_exclusions', {
            'rules': [], 'matches_by_split': {}, 'total_matches': 0,
        }),
        'training_checksum_sha256': entry['training']['checksum_sha256'],
        'development_checksum_sha256': entry['development']['checksum_sha256'],
        'training_examples': entry['training']['examples'],
        'development_examples': entry['development']['examples'],
    }


def load_frozen_training_data(data_release, dataset, *, verify_checksums=True):
    """Load the exact train/development rows shared by every model.

    Paper runs use this function instead of contacting the Hugging Face Hub.  A
    logical row checksum is deliberately independent of Arrow serialization so
    a library upgrade cannot silently create a different experiment identity.
    """

    from datasets import load_from_disk

    root, _, manifest, _ = _read_source_manifest(data_release)
    try:
        entry = manifest['datasets'][dataset]
    except KeyError as error:
        raise ValueError(f'Dataset {dataset!r} is absent from the data release') from error
    training = load_from_disk(str(root / entry['training']['path']))
    development = load_from_disk(str(root / entry['development']['path']))
    if len(training) != entry['training']['examples']:
        raise ValueError(f'Frozen {dataset} training row count changed')
    if len(development) != entry['development']['examples']:
        raise ValueError(f'Frozen {dataset} development row count changed')
    if verify_checksums:
        observed = logical_dataset_checksum(training)
        if observed != entry['training']['checksum_sha256']:
            raise ValueError(f'Frozen {dataset} training checksum changed: {observed}')
        observed = logical_dataset_checksum(development)
        if observed != entry['development']['checksum_sha256']:
            raise ValueError(f'Frozen {dataset} development checksum changed: {observed}')
    return training, development


def load_frozen_development_data(data_release, dataset, *, verify_checksum=True):
    """Load and verify only the frozen development split."""

    from datasets import load_from_disk

    root, _, manifest, _ = _read_source_manifest(data_release)
    try:
        declared = manifest['datasets'][dataset]['development']
    except KeyError as error:
        raise ValueError(f'Dataset {dataset!r} is absent from the data release') from error
    development = load_from_disk(str(root / declared['path']))
    if len(development) != declared['examples']:
        raise ValueError(f'Frozen {dataset} development row count changed')
    if verify_checksum:
        observed = logical_dataset_checksum(development)
        if observed != declared['checksum_sha256']:
            raise ValueError(f'Frozen {dataset} development checksum changed: {observed}')
    return development


def load_frozen_source_split(data_release, dataset, role, split_name,
                             *, verify_checksum=True):
    """Load one development/final source split for evaluation generation."""

    from datasets import load_from_disk

    root, _, manifest, _ = _read_source_manifest(data_release)
    entry = manifest['datasets'][dataset]
    if role == 'development':
        declared = entry['development']
        if declared['source_split'] != split_name:
            raise ValueError(
                f'Frozen {dataset} development split is {declared["source_split"]!r}, '
                f'not {split_name!r}'
            )
    elif role == 'final':
        try:
            declared = entry['final'][split_name]
        except KeyError as error:
            raise ValueError(
                f'Frozen {dataset} final split {split_name!r} is absent'
            ) from error
    else:
        raise ValueError(f'Unsupported source role: {role!r}')
    rows = load_from_disk(str(root / declared['path']))
    if len(rows) != declared['examples']:
        raise ValueError(f'Frozen {dataset}/{role}/{split_name} row count changed')
    if verify_checksum:
        observed = logical_dataset_checksum(rows)
        if observed != declared['checksum_sha256']:
            raise ValueError(
                f'Frozen {dataset}/{role}/{split_name} checksum changed: {observed}'
            )
    return rows


def _manifest_split_for_path(resolved):
    manifest_path = next(
        (parent / 'dataset_manifest.json' for parent in resolved.parents
         if (parent / 'dataset_manifest.json').exists()), None
    )
    if manifest_path is None:
        return None, None, None, None
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    if manifest.get('schema_version', 1) < 2:
        return manifest_path, manifest, None, None
    for dataset, entry in manifest['datasets'].items():
        for key, split in entry['splits'].items():
            if (manifest_path.parent / split['base_path']).resolve() == resolved.resolve():
                return manifest_path, manifest, dataset, (key, split)
    return manifest_path, manifest, None, None


def resolve_eval_base(base, *, split_role='final', split_name=None, purpose='final_evaluation'):
    """Resolve a v1 flat suite or a v2 role-separated Phase 3 suite.

    The explicit purpose check prevents tuning code from accidentally consuming
    labels from an untouched final suite.
    """

    base = Path(base)
    manifest_path = next(
        (
            parent / 'dataset_manifest.json'
            for parent in (base, *base.parents)
            if (parent / 'dataset_manifest.json').exists()
        ),
        None,
    )
    if manifest_path is None:
        if (base / 'original').exists():
            # Compatibility is limited to genuinely pre-manifest flat artifacts.
            return base
        raise FileNotFoundError(f'No evaluation suite or manifest found for {base}')
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    if manifest.get('schema_version', 1) < 2 and (base / 'original').exists():
        return base
    manifest_root = manifest_path.parent
    direct_match = None
    dataset_name = base.name
    for candidate_name, candidate_entry in manifest['datasets'].items():
        for candidate_split in candidate_entry.get('splits', {}).values():
            candidate_path = (manifest_root / candidate_split['base_path']).resolve()
            if candidate_path == base.resolve():
                dataset_name = candidate_name
                direct_match = candidate_split
                break
        if direct_match:
            break
    if dataset_name not in manifest['datasets']:
        raise ValueError(f'Path {base} does not identify a dataset in {manifest_path}')
    entry = manifest['datasets'][dataset_name]
    if direct_match is not None:
        actual_role = direct_match['role']
        if actual_role == 'final' and purpose != 'final_evaluation':
            raise PermissionError(
                'Untouched final labels are unavailable to tuning/development code'
            )
        if split_role != actual_role:
            raise ValueError(
                f'Requested role {split_role!r} does not match direct path role '
                f'{actual_role!r}'
            )
        return base
    if split_role == 'final' and purpose != 'final_evaluation':
        raise PermissionError(
            'Untouched final labels are unavailable to tuning/development code'
        )
    if split_name is None:
        split_name = entry[f'default_{split_role}_split']
    split = entry['splits'].get(f'{split_role}:{split_name}')
    if split is None:
        choices = ', '.join(sorted(entry['splits']))
        raise ValueError(f'Unknown evaluation split {split_role}:{split_name}; {choices}')
    return manifest_root / split['base_path']


def _validate_psych_provenance(transformed, clean, variant):
    """Require explicit psych provenance; raw lexical occurrences are irrelevant."""

    for position, (changed, source) in enumerate(zip(transformed, clean)):
        metadata = changed.get('transform_metadata') or {}
        events = metadata.get('events') or {}
        premise_events = list(events.get('premise') or [])
        hypothesis_events = list(events.get('hypothesis') or [])
        if changed.get('psych_applied') is not True:
            raise ValueError(
                f'Paired set {variant} row {position} lacks psych_applied=True'
            )
        if (
            metadata.get('mode') != 'psych'
            or metadata.get('label_policy') != PSYCH_LABEL_POLICY
            or metadata.get('scope') != 'complete_hypothesis'
        ):
            raise ValueError(f'Paired set {variant} row {position} has invalid psych metadata')
        if premise_events or len(hypothesis_events) != 1:
            raise ValueError(f'Paired set {variant} row {position} has invalid psych provenance')
        event = hypothesis_events[0]
        if (
            event.get('transform') != 'psych'
            or event.get('registry') != 'psych_instruction_inversion'
            or event.get('placement') != 'suffix'
            or event.get('replacement', '').strip() != PSYCH_MARKER
            or int(event.get('output_end', -1)) != len(changed['hypothesis'])
        ):
            raise ValueError(f'Paired set {variant} row {position} has invalid psych suffix event')
        if changed['premise'] != source['premise']:
            raise ValueError(f'Paired set {variant} row {position} changed the premise')
        try:
            restored = invert_from_provenance(changed['hypothesis'], hypothesis_events)
        except (TypeError, ValueError) as error:
            raise ValueError(
                f'Paired set {variant} row {position} has non-invertible psych provenance'
            ) from error
        if restored != source['hypothesis']:
            raise ValueError(f'Paired set {variant} row {position} psych inverse mismatches clean')


def validate_paired_label_policy(transformed, clean, variant, label_policy):
    """Validate paired labels under one declared manifest policy."""

    clean_labels = [int(value) for value in clean['label']]
    transformed_labels = [int(value) for value in transformed['label']]
    if label_policy == LABEL_POLICY_PRESERVE:
        if clean_labels != transformed_labels:
            raise ValueError(f'Paired set label mismatch for {variant} under preserve policy')
        return
    if label_policy == PSYCH_LABEL_POLICY:
        expected = [PSYCH_LABEL_MAP[label] for label in clean_labels]
        if transformed_labels != expected:
            raise ValueError(
                f'Paired set label mismatch for {variant} under {PSYCH_LABEL_POLICY}'
            )
        _validate_psych_provenance(transformed, clean, variant)
        return
    raise ValueError(f'Unknown label policy {label_policy!r} for {variant}')


def load_eval_suite(base, variants=EVAL_VARIANTS, *, split_role='final',
                    split_name=None, purpose='final_evaluation', verify_checksums=True,
                    verify_source_checksum=True):
    """Load transformed sets and their condition-matched clean counterparts."""

    from datasets import load_from_disk

    base = resolve_eval_base(
        base, split_role=split_role, split_name=split_name, purpose=purpose
    )
    manifest_path, manifest, _, split_match = _manifest_split_for_path(base)
    if variants == 'all' or variants == ['all'] or variants == ('all',):
        if split_match is not None:
            variants = tuple(sorted(
                name for name, values in split_match[1]['variants'].items()
                if 'alias_of' not in values
            ))
        else:
            variants = tuple(sorted(
                path.name for path in base.iterdir()
                if path.is_dir() and not path.name.startswith('clean_')
                and path.name != 'original' and (base / f'clean_{path.name}').is_dir()
            ))
    suite = {'original': load_from_disk(str(base / 'original'))}
    if verify_checksums and verify_source_checksum and split_match is not None:
        _, split_entry = split_match
        actual = logical_dataset_checksum(suite['original'])
        if actual != split_entry['source_checksum_sha256']:
            raise ValueError(
                f"Original dataset checksum mismatch for {base}: {actual}"
            )
    for variant in variants:
        declared = split_match[1]['variants'].get(variant) if split_match is not None else None
        if declared is not None:
            manifest_root = next(
                parent for parent in base.parents
                if (parent / 'dataset_manifest.json').exists()
            )
            transformed_path = manifest_root / declared['path']
            paired_clean_path = declared.get('paired_clean_path')
            clean_path = manifest_root / paired_clean_path if paired_clean_path else None
        else:
            transformed_path = base / variant
            clean_path = base / paired_clean_key(variant)
        suite[variant] = load_from_disk(str(transformed_path))
        if clean_path is not None and not clean_path.exists():
            raise FileNotFoundError(
                f"Missing paired clean set {clean_path}; regenerate eval sets with "
                "the Phase 2 prepare_eval_sets.py"
            )
        transformed = suite[variant]
        if clean_path is None:
            if declared.get('paired_clean_storage') != 'source_index_view':
                raise ValueError(f'Unknown paired-clean storage for {variant}')
            indices = [int(value) for value in transformed['source_index']]
            clean = suite['original'].select(indices)
            if getattr(clean, '_indices', None) is not None:
                # Never let Dataset.add_column flatten a view into a cache file
                # beside immutable Arrow data. Materialize this paired view only
                # in process memory.
                clean = clean.flatten_indices(keep_in_memory=True)
            if 'source_index' in clean.column_names:
                clean = clean.remove_columns('source_index')
            clean = clean.add_column('source_index', indices)
        else:
            clean = load_from_disk(str(clean_path))
        if len(clean) != len(transformed):
            raise ValueError(f"Paired set length mismatch for {variant}")
        required = {'source_index', 'label'}
        if not required <= set(clean.column_names) or not required <= set(transformed.column_names):
            raise ValueError(f"Paired set {variant} is missing source_index or label")
        if list(clean['source_index']) != list(transformed['source_index']):
            raise ValueError(f"Paired set source-index mismatch for {variant}")
        if declared is not None and 'label_policy' not in declared:
            if manifest.get('schema_version', 1) >= 4:
                raise ValueError(f'Variant {variant} lacks an explicit label policy')
            label_policy = LABEL_POLICY_PRESERVE
        else:
            label_policy = (
                declared.get('label_policy', LABEL_POLICY_PRESERVE)
                if declared is not None else LABEL_POLICY_PRESERVE
            )
        validate_paired_label_policy(transformed, clean, variant, label_policy)
        suite[paired_clean_key(variant)] = clean
        if verify_checksums and split_match is not None:
            _, split_entry = split_match
            declared = split_entry['variants'].get(variant)
            if declared is None:
                raise ValueError(f'Variant {variant!r} is absent from the dataset manifest')
            transformed_checksum = logical_dataset_checksum(transformed)
            clean_checksum = logical_dataset_checksum(clean)
            if transformed_checksum != declared['checksum_sha256']:
                raise ValueError(f'Transformed checksum mismatch for {variant}')
            if clean_checksum != declared['paired_clean_checksum_sha256']:
                raise ValueError(f'Paired-clean checksum mismatch for {variant}')
    return suite


def verify_frozen_evaluation_data(data_release, datasets=None):
    """Logically verify every declared evaluation row once for a worker."""

    release_root = Path(data_release)
    eval_root = release_root / 'eval_sets'
    manifest_path = eval_root / 'dataset_manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    selected = sorted(datasets or manifest['datasets'])
    unknown = set(selected) - set(manifest['datasets'])
    if unknown:
        raise ValueError(f'Unknown evaluation datasets: {sorted(unknown)}')
    verified = {}
    for dataset in selected:
        verified[dataset] = {}
        for split_key in sorted(manifest['datasets'][dataset]['splits']):
            role, split_name = split_key.split(':', 1)
            purpose = 'final_evaluation' if role == 'final' else 'development'
            split_entry = manifest['datasets'][dataset]['splits'][split_key]
            variants = sorted(
                name for name, declared in split_entry['variants'].items()
                if 'alias_of' not in declared
            )
            source_examples = None
            for index, variant in enumerate(variants):
                suite = load_eval_suite(
                    eval_root / dataset,
                    [variant],
                    split_role=role,
                    split_name=split_name,
                    purpose=purpose,
                    verify_checksums=True,
                    verify_source_checksum=index == 0,
                )
                source_examples = len(suite['original'])
                del suite
            verified[dataset][split_key] = {
                'source_examples': source_examples,
                'variants': len(variants),
            }
    return verified


def eval_suite_identity(base, *, split_role='final', split_name=None,
                        purpose='final_evaluation'):
    """Return the checksummed identity used to prevent cross-split alignment."""

    resolved = resolve_eval_base(
        base, split_role=split_role, split_name=split_name, purpose=purpose
    )
    manifest_path, manifest, dataset_name, split_match = _manifest_split_for_path(resolved)
    if manifest_path is None:
        raise ValueError('Prediction identity requires a Phase 3 dataset manifest')
    if split_match is None:
        raise ValueError(f'Resolved suite {resolved} is absent from its manifest')
    key, split = split_match
    return {
        'config_sha256': manifest['config_sha256'],
        'generation_seed': manifest['seed'],
        'dataset': dataset_name,
        'split_role': split['role'],
        'split_name': split['source_split'],
        'source_checksum_sha256': split['source_checksum_sha256'],
        'split_key': key,
        'variant_checksums': {
            name: {
                'checksum_sha256': values['checksum_sha256'],
                'paired_clean_checksum_sha256': values['paired_clean_checksum_sha256'],
                'label_policy': values.get('label_policy', LABEL_POLICY_PRESERVE),
                'label_changing': values.get('label_changing', False),
                'evaluation_family': values.get(
                    'evaluation_family', 'label_preserving_robustness'
                ),
            }
            for name, values in split['variants'].items()
        },
    }


def write_prediction_manifest(output_dir, base, *, split_role, split_name=None,
                              purpose='final_evaluation', variants=None):
    identity = eval_suite_identity(
        base, split_role=split_role, split_name=split_name, purpose=purpose
    )
    if variants is not None:
        requested = sorted(set(variants) - {'original'} - {
            name for name in variants if name.startswith('clean_')
        })
        missing = set(requested) - set(identity['variant_checksums'])
        if missing:
            raise ValueError(f'Prediction variants absent from manifest: {sorted(missing)}')
        identity['evaluated_variants'] = requested
        identity['variant_checksums'] = {
            name: identity['variant_checksums'][name] for name in requested
        }
    requested_variants = identity.get(
        'evaluated_variants', sorted(identity['variant_checksums'])
    )
    identity['condition_metadata'] = {
        'original': evaluation_condition_metadata('original'),
        **{
            name: evaluation_condition_metadata(
                name,
                label_policy=identity['variant_checksums'][name]['label_policy'],
            )
            for name in requested_variants
        },
        **{
            paired_clean_key(name): evaluation_condition_metadata(
                paired_clean_key(name), label_policy=LABEL_POLICY_PRESERVE
            )
            for name in requested_variants
        },
    }
    path = Path(output_dir) / 'predictions_manifest.json'
    path.write_text(json.dumps(identity, indent=2) + '\n', encoding='utf-8')
    return identity


def select_training_and_development(
    raw, dataset, *, fraction=0.05, seed=DEFAULT_DATA_SPLIT_SEED
):
    """Return training and tuning data without touching final evaluation labels."""

    if dataset == 'snli':
        return raw['train'], raw['validation']
    if dataset != 'multi_nli':
        raise ValueError(f'Unsupported NLI dataset: {dataset}')
    split = raw['train'].train_test_split(
        test_size=fraction, seed=seed, stratify_by_column='label'
    )
    return split['train'], split['test']


def robustness_drops(results, variants=None):
    """Compute each robustness drop against its exactly paired clean subset."""

    if variants is None:
        variants = tuple(
            name for name in results
            if name != 'original' and not name.startswith('clean_')
            and paired_clean_key(name) in results
            and evaluation_condition_metadata(
                name,
                label_policy=results[name].get('condition_metadata', {}).get(
                    'label_policy'
                ),
            )['label_changing'] is False
        )

    return {
        variant: (
            results[paired_clean_key(variant)]['accuracy']
            - results[variant]['accuracy']
        ) * 100
        for variant in variants
        if evaluation_condition_metadata(
            variant,
            label_policy=results[variant].get('condition_metadata', {}).get(
                'label_policy'
            ),
        )['label_changing'] is False
    }
