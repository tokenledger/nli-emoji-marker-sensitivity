"""Shared fast/restartable runtime helpers for model pipelines."""

from __future__ import annotations

import json
from pathlib import Path


TOKENIZER_BACKENDS = {'fast', 'slow'}
TOKENIZER_NORMALIZATION = {'default', 'enabled', 'disabled', 'spm_byte_fallback'}
# 'spm_byte_fallback': load the fast tokenizer with library defaults, then force
# the converted Unigram model to honour SentencePiece byte fallback. Needed for
# microsoft/deberta-v3-base, whose Transformers conversion drops byte_fallback and
# maps five principal emoji to [UNK]. 'enabled'/'disabled' remain the BERTweet
# `normalization=` constructor flag and are unchanged.
MIXED_PRECISION_MODES = {'auto', 'no', 'fp16', 'bf16'}


def load_tokenizer(model_name: str, revision: str | None = None, *,
                   backend: str = 'fast', normalization: str = 'default'):
    """Load a tokenizer using the frozen model-specific input contract."""

    if backend not in TOKENIZER_BACKENDS:
        raise ValueError(f'Unknown tokenizer backend: {backend!r}')
    if normalization not in TOKENIZER_NORMALIZATION:
        raise ValueError(f'Unknown tokenizer normalization: {normalization!r}')
    from transformers import AutoTokenizer

    kwargs = {
        'use_fast': backend == 'fast',
        **pretrained_kwargs(revision),
    }
    if normalization in {'enabled', 'disabled'}:
        kwargs['normalization'] = normalization == 'enabled'
    tokenizer = AutoTokenizer.from_pretrained(model_name, **kwargs)
    if normalization == 'spm_byte_fallback':
        tokenizer = apply_spm_byte_fallback(tokenizer)
    return tokenizer


def unigram_byte_fallback_state(tokenizer) -> bool | None:
    """Return the backend Unigram model's byte_fallback flag, or None if absent."""

    backend = getattr(tokenizer, 'backend_tokenizer', None)
    if backend is None:
        return None
    model = json.loads(backend.to_str()).get('model') or {}
    if model.get('type') != 'Unigram':
        return None
    return bool(model.get('byte_fallback', False))


def apply_spm_byte_fallback(tokenizer):
    """Force byte fallback on a fast tokenizer converted from a SentencePiece Unigram model.

    The tokenizers Python API exposes no setter for ``Unigram.byte_fallback``, so
    the backend is re-created from its JSON with ``byte_fallback: true`` and
    reattached. Idempotent: a tokenizer whose saved ``tokenizer.json`` already
    carries the flag is returned unchanged. Fails loudly for any tokenizer that is
    not a fast Unigram tokenizer, so the contract can never silently degrade to
    the [UNK]-collapsing default.
    """

    from tokenizers import Tokenizer

    if not getattr(tokenizer, 'is_fast', False) or getattr(tokenizer, 'backend_tokenizer', None) is None:
        raise RuntimeError(
            f'spm_byte_fallback requires a fast tokenizer with a backend model; got '
            f'{type(tokenizer).__name__}'
        )
    payload = json.loads(tokenizer.backend_tokenizer.to_str())
    model = payload.get('model') or {}
    if model.get('type') != 'Unigram':
        raise RuntimeError(
            f'spm_byte_fallback requires a Unigram (SentencePiece) backend model; got '
            f'{model.get("type")!r} for {type(tokenizer).__name__}'
        )
    if model.get('byte_fallback') is True:
        return tokenizer
    model['byte_fallback'] = True
    payload['model'] = model
    patched = Tokenizer.from_str(json.dumps(payload))
    if patched.get_vocab_size(with_added_tokens=True) != tokenizer.backend_tokenizer.get_vocab_size(with_added_tokens=True):
        raise RuntimeError('spm_byte_fallback patch changed the tokenizer vocabulary size')
    tokenizer._tokenizer = patched
    if unigram_byte_fallback_state(tokenizer) is not True:
        raise RuntimeError('spm_byte_fallback patch did not take effect on the backend model')
    return tokenizer


def pretrained_kwargs(revision: str | None) -> dict:
    """Kwargs for tokenizer loading (revision only; tokenizers take no dtype)."""

    return {'revision': revision} if revision else {}


MODEL_PARAMETER_DTYPE = 'float32'


def model_pretrained_kwargs(revision: str | None, *, dtype: str = MODEL_PARAMETER_DTYPE) -> dict:
    """Kwargs for model loading: pinned revision plus explicit fp32 master weights.

    transformers 5.x defaults ``from_pretrained`` to ``dtype='auto'``, which
    instantiates the model (including any freshly initialised classifier head)
    in the checkpoint's storage dtype. microsoft/deberta-v3-base is stored in
    fp16, so without this the optimizer ran on fp16 master weights and the SNLI
    fine-tune collapsed to the majority class. Trainer bf16/fp16 autocast still
    applies on top of fp32 parameters, which is the intended regime.
    """

    import torch

    return {**pretrained_kwargs(revision), 'dtype': getattr(torch, dtype)}


def assert_float32_parameters(model, label: str = 'model') -> str:
    """Raise unless every parameter is fp32; return the resolved dtype string."""

    import torch

    offending = {
        name: str(parameter.dtype)
        for name, parameter in model.named_parameters()
        if parameter.dtype != torch.float32
    }
    if offending:
        sample = dict(list(offending.items())[:5])
        raise RuntimeError(
            f'{label} has {len(offending)} non-fp32 parameters (e.g. {sample}); '
            'fine-tuning requires fp32 master weights. Load with '
            'model_pretrained_kwargs(...) so dtype=torch.float32 is explicit.'
        )
    return str(torch.float32)


def tokenization_cache_operation(
    tokenizer_name: str,
    revision: str | None,
    max_length: int,
    purpose: str,
    *,
    tokenizer_backend: str = 'fast',
    tokenizer_normalization: str = 'default',
    input_preprocessing: str = 'none',
) -> str:
    """Describe every input that can change cached token IDs."""

    return 'tokenize:' + json.dumps({
        'tokenizer': tokenizer_name,
        'revision': revision,
        'backend': tokenizer_backend,
        'normalization': tokenizer_normalization,
        'input_preprocessing': input_preprocessing,
        'max_length': int(max_length),
        'purpose': purpose,
    }, sort_keys=True, separators=(',', ':'))


def mixed_precision_kwargs(torch_module, mode: str = 'auto') -> dict[str, bool]:
    """Resolve an explicit or hardware-aware Trainer precision policy."""

    if mode not in MIXED_PRECISION_MODES:
        raise ValueError(f'Unknown mixed precision mode: {mode!r}')
    cuda_available = bool(torch_module.cuda.is_available())
    if mode == 'no' or not cuda_available:
        if mode in {'fp16', 'bf16'} and not cuda_available:
            raise RuntimeError(f'{mode} mixed precision requires a CUDA device')
        return {'fp16': False, 'bf16': False}
    if mode == 'bf16':
        is_supported = getattr(torch_module.cuda, 'is_bf16_supported', lambda: False)
        if not is_supported():
            raise RuntimeError('bf16 mixed precision is not supported by this CUDA device')
        return {'fp16': False, 'bf16': True}
    if mode == 'fp16':
        return {'fp16': True, 'bf16': False}
    is_supported = getattr(torch_module.cuda, 'is_bf16_supported', lambda: False)
    use_bf16 = bool(is_supported())
    return {'fp16': not use_bf16, 'bf16': use_bf16}


def resume_checkpoint(output_dir, audited_dataset=None):
    """Return the last complete epoch checkpoint and restore audit counters."""

    from transformers.trainer_utils import get_last_checkpoint

    output = Path(output_dir)
    checkpoint = get_last_checkpoint(str(output)) if output.is_dir() else None
    if not checkpoint:
        return None
    state_path = Path(checkpoint) / 'trainer_state.json'
    if not state_path.is_file():
        raise RuntimeError(f'Resume checkpoint is missing trainer_state.json: {checkpoint}')
    state = json.loads(state_path.read_text(encoding='utf-8'))
    epoch = float(state.get('epoch') or 0.0)
    completed_epochs = int(epoch)
    if abs(epoch - completed_epochs) > 1e-6:
        raise RuntimeError(
            f'Refusing partial-epoch resume at epoch {epoch}; paper runs checkpoint '
            'only at complete epoch boundaries.'
        )
    if audited_dataset is not None and hasattr(audited_dataset, 'assume_completed_epochs'):
        audited_dataset.assume_completed_epochs(completed_epochs)
    print(f'Resuming from verified epoch {completed_epochs} checkpoint: {checkpoint}')
    return checkpoint


def model_source_metadata(model_name: str, revision: str | None, *,
                          tokenizer_backend: str = 'fast',
                          tokenizer_normalization: str = 'default',
                          input_preprocessing: str = 'none',
                          parameter_dtype: str | None = None) -> dict:
    metadata = {
        'model_repository': model_name,
        'model_revision': revision,
        'tokenizer_repository': model_name,
        'tokenizer_revision': revision,
        'tokenizer_backend': tokenizer_backend,
        'tokenizer_normalization': tokenizer_normalization,
        'input_preprocessing': input_preprocessing,
    }
    if parameter_dtype is not None:
        metadata['parameter_dtype'] = parameter_dtype
    return metadata
