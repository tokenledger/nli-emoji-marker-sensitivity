"""Exact McNemar and Holm helpers shared by the two inference-only analyses."""
from __future__ import annotations
import numpy as np
from scipy.stats import binomtest


def mcnemar_exact(a_event: np.ndarray, b_event: np.ndarray) -> dict:
    """Two-sided exact McNemar on paired binary events (same sources). Returns b, c, p."""
    a = np.asarray(a_event, dtype=bool)
    b = np.asarray(b_event, dtype=bool)
    only_a = int(np.sum(a & ~b))
    only_b = int(np.sum(~a & b))
    n = only_a + only_b
    p = 1.0 if n == 0 else float(binomtest(min(only_a, only_b), n, 0.5, alternative="two-sided").pvalue)
    return {"only_a": only_a, "only_b": only_b, "discordant": n, "p": p}


def holm(pvalues: list[float]) -> list[float]:
    """Holm step-down adjusted p-values (monotone, capped at 1)."""
    m = len(pvalues)
    order = sorted(range(m), key=lambda i: pvalues[i])
    adjusted = [0.0] * m
    running = 0.0
    for rank, i in enumerate(order):
        val = min(1.0, (m - rank) * pvalues[i])
        running = max(running, val)
        adjusted[i] = running
    return adjusted
