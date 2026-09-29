"""Machine-readable and human-readable Hugging Face Trainer progress logging."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import time

from transformers import TrainerCallback


class ProgressCallback(TrainerCallback):
    """Write resumable progress state and print concise ETA-bearing events."""

    def __init__(self, output_dir: str | Path):
        self.output_dir = Path(output_dir)
        self.started = None
        self.started_step = 0

    def _write(self, event: str, state, values=None):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        now = time.monotonic()
        values = dict(values or {})
        total = int(state.max_steps or 0)
        step = int(state.global_step or 0)
        elapsed = (now - self.started) if self.started is not None else 0.0
        completed_here = max(0, step - self.started_step)
        rate = completed_here / elapsed if completed_here and elapsed else 0.0
        eta = (total - step) / rate if rate and total >= step else None
        record = {
            "event": event,
            "time_utc": datetime.now(timezone.utc).isoformat(),
            "step": step,
            "total_steps": total,
            "percent": round(100.0 * step / total, 2) if total else None,
            "epoch": state.epoch,
            "elapsed_seconds": round(elapsed, 1),
            "eta_seconds": round(eta, 1) if eta is not None else None,
            **values,
        }
        line = json.dumps(record, sort_keys=True)
        with (self.output_dir / "progress.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
        temporary = self.output_dir / ".progress.json.tmp"
        temporary.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
        temporary.replace(self.output_dir / "progress.json")
        print(
            f"PROGRESS step={step}/{total} ({record['percent']}%) "
            f"epoch={state.epoch} eta_s={record['eta_seconds']} event={event}",
            flush=True,
        )

    def on_train_begin(self, args, state, control, **kwargs):
        del args, control, kwargs
        self.started = time.monotonic()
        self.started_step = int(state.global_step or 0)
        self._write("train_begin", state)

    def on_log(self, args, state, control, logs=None, **kwargs):
        del args, control, kwargs
        self._write("log", state, logs)

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        del args, control, kwargs
        self._write("evaluate", state, metrics)

    def on_train_end(self, args, state, control, **kwargs):
        del args, control, kwargs
        self._write("train_end", state)
