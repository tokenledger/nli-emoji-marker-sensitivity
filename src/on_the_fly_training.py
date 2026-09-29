"""Compute-matched, epoch-aware marker augmentation for Trainer datasets."""

from __future__ import annotations

import hashlib
import tempfile

from torch.utils.data import Dataset
from transformers import TrainerCallback

from preprocessing import preprocess_example
from transforms import apply_training_condition
from helpers import map_dataset_locally, prepare_dataset_nli
from training_runtime import tokenization_cache_operation


PRESENTATION_SEED_MIXER = 'blake2b-64-v1(run_seed,epoch,index)'


def presentation_seed(run_seed, epoch, index):
    """Mix a presentation identity without shifted streams across run seeds."""

    payload = f'{int(run_seed)}\0{int(epoch)}\0{int(index)}'.encode('ascii')
    digest = hashlib.blake2b(
        payload, digest_size=8, person=b'nli-aug-v1'
    ).digest()
    return int.from_bytes(digest, byteorder='big', signed=False)


class _AuditedPresentationDataset(Dataset):
    """Track which source indices a single-process Trainer presents per epoch."""

    audit_policy = "one presentation of each clean example per epoch"

    def _initialize_audit(self):
        self.epoch = 0
        self._seen_by_epoch = {}
        self._unique_presentations = 0
        self._total_fetches = 0
        self._duplicate_presentations = 0
        self._transformed_presentations = 0

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def assume_completed_epochs(self, completed_epochs):
        """Restore epoch-boundary audit state when Trainer resumes a checkpoint."""

        for epoch in range(int(completed_epochs)):
            if epoch in self._seen_by_epoch:
                continue
            self._seen_by_epoch[epoch] = bytearray(b'\x01') * len(self)
            self._unique_presentations += len(self)
            self._total_fetches += len(self)

    def _record_presentation(self, index, was_transformed=False):
        seen = self._seen_by_epoch.setdefault(self.epoch, bytearray(len(self)))
        self._total_fetches += 1
        if seen[index]:
            self._duplicate_presentations += 1
        else:
            seen[index] = 1
            self._unique_presentations += 1
        self._transformed_presentations += int(was_transformed)

    def audit(self, completed_epochs):
        expected = len(self) * int(completed_epochs)
        return {
            "policy": self.audit_policy,
            "dataset_examples": len(self),
            "completed_epochs": int(completed_epochs),
            "expected_presentations": expected,
            "unique_presentations": self._unique_presentations,
            "total_fetches": self._total_fetches,
            "duplicate_presentations": self._duplicate_presentations,
            "missing_presentations": max(0, expected - self._unique_presentations),
            "transformed_presentations": self._transformed_presentations,
            "observed": True,
        }


class AuditedCleanDataset(_AuditedPresentationDataset):
    """Expose a tokenized clean dataset while observing actual Trainer fetches."""

    def __init__(self, source):
        self.source = source
        self._initialize_audit()

    def __len__(self):
        return len(self.source)

    def __getitem__(self, index):
        index = int(index)
        self._record_presentation(index)
        return dict(self.source[index])


class PreparedEpochDataset(_AuditedPresentationDataset):
    """Serve batched/pretokenized presentations selected by Trainer epoch."""

    audit_policy = (
        "one presentation of each clean example per epoch with precomputed transformation"
    )

    def __init__(self, epochs, *, cache_owner=None):
        if not epochs:
            raise ValueError('At least one prepared epoch is required')
        lengths = {len(dataset) for dataset in epochs}
        if len(lengths) != 1:
            raise ValueError('Prepared epoch datasets must have identical lengths')
        self.epochs = tuple(epochs)
        self.source = self.epochs[0]
        # When no cache path is supplied, keep the temporary on-disk Arrow cache
        # alive for exactly as long as this memory-mapped training dataset.
        self._cache_owner = cache_owner
        self._initialize_audit()

    def __len__(self):
        return len(self.source)

    def close(self):
        cache_owner = getattr(self, '_cache_owner', None)
        if cache_owner is not None:
            cache_owner.cleanup()
            self._cache_owner = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            # Interpreter shutdown may have already torn down tempfile helpers.
            pass

    def __getitem__(self, index):
        index = int(index)
        epoch = min(self.epoch, len(self.epochs) - 1)
        encoded = dict(self.epochs[epoch][index])
        transformed = bool(encoded.pop('_was_transformed', False))
        self._record_presentation(index, transformed)
        return encoded

    def assume_completed_epochs(self, completed_epochs):
        completed_epochs = int(completed_epochs)
        super().assume_completed_epochs(completed_epochs)
        for epoch in range(min(completed_epochs, len(self.epochs))):
            flags = self.epochs[epoch]['_was_transformed']
            self._transformed_presentations += sum(bool(value) for value in flags)


def prepare_epoch_tokenized_dataset(
    source, tokenizer, max_length, *, epochs, augmentation_probability,
    marker_placements, seed, cache_dir=None, num_proc=1,
    preprocess_mode=None, marker_choices=None, input_preprocessing='none',
    model_revision=None, tokenizer_backend='fast',
    tokenizer_normalization='default',
):
    """Batch-generate/tokenize every epoch once, then keep the GPU fed quickly."""

    prepared_epochs = []
    cache_owner = None
    working_cache_dir = cache_dir
    if working_cache_dir is None:
        cache_owner = tempfile.TemporaryDirectory(prefix='nli_epoch_cache_')
        working_cache_dir = cache_owner.name
    marker_choices = tuple(marker_choices or ())
    marker_placements = tuple(marker_placements)
    for epoch in range(int(epochs)):
        def transform_one(example, index, *, _epoch=epoch):
            item = dict(example)
            if preprocess_mode is not None:
                item = preprocess_example(item, mode=preprocess_mode)
            transformed = apply_training_condition(
                item,
                augmentation_probability=augmentation_probability,
                marker_placements=marker_placements,
                marker_choices=marker_choices or None,
                seed=presentation_seed(seed, _epoch, index),
                include_metadata=True,
            )
            events = transformed.pop('transform_metadata')['events']
            transformed['_was_transformed'] = any(
                events[field] for field in ('premise', 'hypothesis')
            )
            return transformed

        condition_key = (
            f'presentations:seed={seed}:epoch={epoch}:ratio={augmentation_probability}:'
            f'placements={marker_placements}:markers={marker_choices}:'
            f'preprocess={preprocess_mode}:model-input={input_preprocessing}:'
            f'seed-mixer={PRESENTATION_SEED_MIXER}'
        )
        text_rows = map_dataset_locally(
            source, transform_one, cache_dir=working_cache_dir, operation=condition_key,
            remove_columns=source.column_names, with_indices=True,
            num_proc=num_proc, desc=f'Preparing training epoch {epoch + 1}/{epochs}',
        )

        def tokenize_batch(examples):
            encoded = prepare_dataset_nli(
                examples, tokenizer, max_length, input_preprocessing
            )
            encoded['_was_transformed'] = examples['_was_transformed']
            return encoded

        tokenized = map_dataset_locally(
            text_rows, tokenize_batch, cache_dir=working_cache_dir,
            operation=tokenization_cache_operation(
                tokenizer.name_or_path, model_revision, max_length, condition_key,
                tokenizer_backend=tokenizer_backend,
                tokenizer_normalization=tokenizer_normalization,
                input_preprocessing=input_preprocessing,
            ),
            batched=True, remove_columns=text_rows.column_names,
            num_proc=num_proc, desc=f'Tokenizing training epoch {epoch + 1}/{epochs}',
        )
        prepared_epochs.append(tokenized)
    return PreparedEpochDataset(prepared_epochs, cache_owner=cache_owner)


class OnTheFlyMarkerDataset(_AuditedPresentationDataset):
    """Transform each clean row when it is fetched, once per sampler epoch.

    The transformation seed is a pure function of the experiment seed, epoch,
    and source index. This makes runs reproducible while giving each epoch a new
    presentation and retaining exactly the baseline dataset length.
    """

    def __init__(self, source, tokenizer, max_length, *, augmentation_probability,
                 marker_placements, seed, preprocess_mode=None, marker_choices=None,
                 input_preprocessing='none'):
        self.source = source
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.augmentation_probability = augmentation_probability
        self.marker_placements = tuple(marker_placements)
        self.seed = int(seed)
        self.preprocess_mode = preprocess_mode
        self.marker_choices = tuple(marker_choices or ())
        self.input_preprocessing = input_preprocessing
        self._initialize_audit()

    def __len__(self):
        return len(self.source)

    def assume_completed_epochs(self, completed_epochs):
        completed_epochs = int(completed_epochs)
        super().assume_completed_epochs(completed_epochs)
        # Checkpoints are written on epoch boundaries. Reconstruct only the
        # small audit counter; no model tokenization is performed here.
        original_epoch = self.epoch
        for epoch in range(completed_epochs):
            self.epoch = epoch
            for index in range(len(self)):
                _, transformed = self._presentation(index)
                self._transformed_presentations += int(transformed)
        self.epoch = original_epoch

    def _presentation(self, index):
        example = dict(self.source[int(index)])
        if self.preprocess_mode is not None:
            example = preprocess_example(example, mode=self.preprocess_mode)
        transformed = apply_training_condition(
            example,
            augmentation_probability=self.augmentation_probability,
            marker_placements=self.marker_placements,
            marker_choices=self.marker_choices or None,
            seed=presentation_seed(self.seed, self.epoch, index),
            include_metadata=True,
        )
        events = transformed["transform_metadata"]["events"]
        was_transformed = any(events[field] for field in ("premise", "hypothesis"))
        transformed.pop("transform_metadata")
        return transformed, was_transformed

    def __getitem__(self, index):
        index = int(index)
        example, was_transformed = self._presentation(index)
        self._record_presentation(index, was_transformed)
        encoded = prepare_dataset_nli(
            example, self.tokenizer, self.max_length, self.input_preprocessing
        )
        return dict(encoded)

    audit_policy = (
        "one presentation of each clean example per epoch with on-the-fly transformation"
    )


class SetDatasetEpochCallback(TrainerCallback):
    """Keep the dataset's deterministic presentation epoch aligned to Trainer."""

    def on_epoch_begin(self, args, state, control, train_dataloader=None, **kwargs):
        del args, control, kwargs
        dataset = getattr(train_dataloader, "dataset", None)
        if hasattr(dataset, "set_epoch"):
            dataset.set_epoch(int(state.epoch or 0))


def validate_training_audit(audit):
    """Fail a completed run if its compute-matching invariant was violated."""

    if audit["duplicate_presentations"]:
        raise RuntimeError(
            f"Training fetched {audit['duplicate_presentations']} duplicate example/epoch presentations"
        )
    if audit["missing_presentations"]:
        raise RuntimeError(
            f"Training missed {audit['missing_presentations']} planned example/epoch presentations"
        )


def validate_training_world_size(world_size):
    """Reject DDP before training until audit counters are reduced across ranks."""

    if int(world_size) != 1:
        raise RuntimeError(
            "Phase 4 presentation auditing currently requires world_size=1; "
            "distributed sampler padding and rank aggregation are not implemented"
        )
