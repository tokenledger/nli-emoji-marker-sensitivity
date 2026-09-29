"""Input and output locations of the analysis module.

Every analysis script resolves its files through this module. The locations
come from three environment variables, which the Makefile sets:

  PREDICTIONS_DIR  saved predictions (layout in analysis/README.md)
  DATA_DIR         data directory written by the dataprep module
  OUTPUT_DIR       destination of tables, reports, and figures
"""

from __future__ import annotations

import os
from pathlib import Path


def _root(variable: str) -> Path:
    value = os.environ.get(variable)
    if not value:
        raise SystemExit(
            f'{variable} is not set. Run the analysis through its Makefile, or '
            f'export {variable}; see analysis/README.md.'
        )
    return Path(value).expanduser().resolve()


def _existing(variable: str, parts: tuple[str, ...]) -> Path:
    path = _root(variable).joinpath(*parts)
    if not path.exists():
        raise SystemExit(
            f'Required input is missing: {path} ({variable}={_root(variable)}). '
            'The expected layout is described in analysis/README.md.'
        )
    return path


def predictions(*parts: str) -> Path:
    """Return an existing path below PREDICTIONS_DIR."""

    return _existing('PREDICTIONS_DIR', parts)


def data(*parts: str) -> Path:
    """Return an existing path below DATA_DIR."""

    return _existing('DATA_DIR', parts)


def output(*parts: str) -> Path:
    """Return a directory below OUTPUT_DIR, created when absent."""

    path = _root('OUTPUT_DIR').joinpath(*parts)
    path.mkdir(parents=True, exist_ok=True)
    return path


def result(*parts: str) -> Path:
    """Return an existing file written to OUTPUT_DIR by an earlier target."""

    path = _root('OUTPUT_DIR').joinpath(*parts)
    if not path.exists():
        raise SystemExit(
            f'Required result is missing: {path}. It is written by an earlier '
            'make target; see the dependency notes in analysis/README.md.'
        )
    return path


def checkpoints(*parts: str) -> Path:
    """Return a path below CHECKPOINTS_DIR (fine-tuned seed-42 checkpoints)."""

    value = os.environ.get('CHECKPOINTS_DIR')
    if not value:
        raise FileNotFoundError(
            'CHECKPOINTS_DIR is not set; it must hold the fine-tuned checkpoints '
            '(layout in analysis/README.md)'
        )
    return Path(value).expanduser().resolve().joinpath(*parts)
