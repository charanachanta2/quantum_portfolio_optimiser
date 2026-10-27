import numpy as np
from stock_prediction.fuzzy import fuzzy_risk, fuzzy_signal, trap
from stock_prediction.ga import real_ga, decode, BOUNDS, NAMES


def test_trap_shapes():
    assert trap(5, 0, 2, 8, 10) == 1.0 and trap(1, 0, 2, 8, 10) == 0.5 and trap(11, 0, 2, 8, 10) == 0.0


def test_readme_risk_rule_and_monotonicity():
    hi, _ = fuzzy_risk(0.60, 0.0, 0.1)               # IF volatility high AND confidence low THEN risk high
    lo, _ = fuzzy_risk(0.08, -0.5, 0.9)
    assert hi > 0.75 > 0.25 > lo
    scores = [fuzzy_risk(v, 0.0, 0.3)[0] for v in (0.08, 0.2, 0.3, 0.5, 0.8)]
    assert all(b >= a - 1e-9 for a, b in zip(scores, scores[1:]))
    assert fuzzy_risk(0.2, 3.0, 0.5)[0] > fuzzy_risk(0.2, -1.0, 0.5)[0]      # VIX stress raises risk
    for v in (0.1, 0.25, 0.4):                                              # lower confidence never lowers risk
        cs = [fuzzy_risk(v, 0.5, c)[0] for c in (0.0, 0.2, 0.4, 0.6, 1.0)]
        assert all(b <= a + 1e-9 for a, b in zip(cs, cs[1:]))
    assert fuzzy_risk(0.25, 0.5, 0.5)[1] in ("Low Risk", "Medium Risk", "High Risk")


def test_readme_buy_rule():
    s, lab = fuzzy_signal(6.0, trend=0.08, volume_ratio=2.0, risk=0.2)       # IF volume high AND trend rising THEN strong buy
    assert lab == "STRONG BUY" and s > 0.6
    assert fuzzy_signal(-6.0, -0.08, 2.0, 0.9)[1] == "STRONG SELL"
    assert fuzzy_signal(0.0, 0.0, 1.0, 0.5)[1] == "HOLD"


def test_ga_minimises_and_respects_bounds():
    target = np.array([1e-3, 64, 0.2, 128, 40.0])
    lo = np.array([BOUNDS[k][0] for k in NAMES]); hi = np.array([BOUNDS[k][1] for k in NAMES])
    f = lambda c: float((((c - target) / (hi - lo)) ** 2).sum())
    best, fit, hist = real_ga(f, pop=14, gens=25, seed=0)
    assert fit < 0.02 and hist[-1] <= hist[0] and (best >= lo).all() and (best <= hi).all()
    d = decode(best); assert set(d) == {"lr", "units", "dropout", "batch", "seq_len"} and d["units"] % 8 == 0
