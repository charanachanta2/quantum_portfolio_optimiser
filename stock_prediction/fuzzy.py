"""Fuzzy inference (E1).  Zero-order Takagi-Sugeno systems with triangular/trapezoidal memberships:
rule firing strength = min over antecedents (AND), output = firing-strength-weighted mean of rule consequents.
Rules follow the README examples:
    IF Volume High AND Trend Rising THEN Buy is Strong
    IF Volatility High AND Confidence Low THEN Risk is High
"""
from __future__ import annotations
import numpy as np


def trap(x, a, b, c, d):
    """Trapezoid membership (a<=b<=c<=d); use b==c for a triangle, a==b / c==d for shoulders."""
    x = float(x)
    if x < a or x > d:
        return 0.0
    if b <= x <= c:
        return 1.0
    if x < b:
        return (x - a) / (b - a) if b > a else 1.0
    return (d - x) / (d - c) if d > c else 1.0


def _vol_sets(ann_vol):            # annualised realised volatility
    return {"low": trap(ann_vol, -1, -1, 0.12, 0.22), "med": trap(ann_vol, 0.12, 0.22, 0.30, 0.42),
            "high": trap(ann_vol, 0.30, 0.45, 5, 5)}


def _z_sets(z):                    # VIX z-score vs trailing year
    return {"calm": trap(z, -9, -9, -0.3, 0.7), "elev": trap(z, -0.3, 0.7, 1.0, 2.0), "stress": trap(z, 1.0, 2.0, 9, 9)}


def _conf_sets(c):                 # model confidence in [0,1]
    return {"low": trap(c, -1, -1, 0.25, 0.55), "high": trap(c, 0.25, 0.55, 2, 2)}


# consequents of the 3x3 rule grid  (rows: volatility low/med/high, columns: VIX calm/elevated/stress); monotone in both inputs
RISK_GRID = {("low", "calm"): 0.12, ("low", "elev"): 0.35, ("low", "stress"): 0.70,
             ("med", "calm"): 0.40, ("med", "elev"): 0.60, ("med", "stress"): 0.85,
             ("high", "calm"): 0.70, ("high", "elev"): 0.85, ("high", "stress"): 0.95}
CONF_PENALTY = 0.35        # low forecast confidence closes up to 35 % of the gap to "maximum risk"


def fuzzy_risk(ann_vol: float, vix_z: float = 0.0, confidence: float = 0.5) -> tuple[float, str]:
    """-> (risk in [0,1], label).  0 = very low risk, 1 = very high.
    Rules IF vol is V AND VIX is Z THEN risk is R (grid above), then  IF confidence is Low THEN risk moves toward High
    (so 'volatility high AND confidence low -> risk high' holds, and lower confidence can never LOWER the risk)."""
    v, z, c = _vol_sets(ann_vol), _z_sets(vix_z), _conf_sets(confidence)
    num = den = 0.0
    for (vi, zi), out in RISK_GRID.items():
        w = min(v[vi], z[zi])
        num += w * out; den += w
    base = num / den if den > 1e-9 else 0.5
    r = base + (1.0 - base) * CONF_PENALTY * c["low"]
    return float(r), ("Low Risk" if r < 0.33 else "Medium Risk" if r < 0.66 else "High Risk")


def _ret_sets(pct):                # expected horizon return, percent
    return {"neg": trap(pct, -99, -99, -3, -0.5), "flat": trap(pct, -3, -0.5, 0.5, 3), "pos": trap(pct, 0.5, 3, 99, 99)}


def fuzzy_signal(exp_ret_pct: float, trend: float = 0.0, volume_ratio: float = 1.0, risk: float = 0.5) -> tuple[float, str]:
    """-> (score in [-1,1], label).  trend = price/MA20-1 (rising > 0); volume_ratio = volume / 20d average (1 if unknown)."""
    e = _ret_sets(exp_ret_pct)
    t = {"down": trap(trend, -1, -1, -0.04, 0.0), "up": trap(trend, 0.0, 0.04, 1, 1)}
    vol = {"high": trap(volume_ratio, 1.0, 1.5, 9, 9), "norm": trap(volume_ratio, 0, 0.5, 1.2, 1.6)}
    rk = {"high": trap(risk, 0.5, 0.8, 2, 2), "low": trap(risk, -1, -1, 0.3, 0.6)}
    rules = [
        (min(vol["high"], t["up"], e["pos"]), 1.0),   # README rule: high volume AND rising -> strong buy
        (min(e["pos"], rk["low"]), 0.7),
        (min(e["pos"], rk["high"]), 0.3),
        (min(e["flat"], t["up"]), 0.2),
        (e["flat"], 0.0),
        (min(e["flat"], t["down"]), -0.2),
        (min(e["neg"], rk["high"]), -1.0),
        (min(e["neg"], t["down"], vol["high"]), -1.0),
        (e["neg"], -0.6),
    ]
    num = sum(w * o for w, o in rules)
    den = sum(w for w, _ in rules)
    s = num / den if den > 1e-9 else 0.0
    lab = ("STRONG BUY" if s > 0.6 else "BUY" if s > 0.25 else "HOLD" if s > -0.25 else "SELL" if s > -0.6 else "STRONG SELL")
    return float(s), lab
