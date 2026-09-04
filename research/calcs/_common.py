"""Shared parameters and helpers for the LEAPS research calculations.

Baseline calibration (a generic large-cap US name, mid-2026 conditions):
    S0 = 100, sigma = 25%, r = 4.2%, q = 1.5%
"""
from __future__ import annotations

import math
from pathlib import Path

import bs

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "research" / "results"
OUT.mkdir(parents=True, exist_ok=True)

S0 = 100.0
SIG = 0.25
R = 0.042
Q = 0.015
DELTA_TGT = 0.80


def strike_for_delta(S, T, r, q, sig, target, kind="C"):
    """Strike such that |delta| = target (bisection; delta is monotone in K)."""
    f = (lambda K: bs.delta_call(S, K, T, r, q, sig) - target) if kind == "C" \
        else (lambda K: -bs.delta_put(S, K, T, r, q, sig) - target)
    a, b = 1e-4, 20.0 * S
    fa = f(a)
    for _ in range(200):
        m = 0.5 * (a + b)
        if fa * f(m) <= 0:
            b = m
        else:
            a, fa = m, f(m)
    return 0.5 * (a + b)
