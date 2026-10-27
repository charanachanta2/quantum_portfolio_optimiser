import numpy as np, pandas as pd
from scripts.build_esg import risk_to_score


def test_risk_scores_are_flipped_and_percentile_based():
    risk = pd.Series([12., 18., 25., 30., 35., 40.])            # Sustainalytics: LOWER = better
    sc = risk_to_score(risk, higher_is_better=False)
    assert sc.is_monotonic_decreasing and sc.iloc[0] > 85 and sc.iloc[-1] < 15
    assert ((sc >= 40).mean()) < 1.0                              # the esg_min=40 filter actually removes names
    assert np.allclose(risk_to_score(risk, True).values, sc.values[::-1])


# --- added: guards for hand-collected score files (S&P / MSCI lookups) ---
import sys, pytest
from scripts import build_esg
from qpo.universe import UNIVERSES


def _run(monkeypatch, tmp_path, rows, *extra):
    f = tmp_path / "s.csv"
    pd.DataFrame(rows, columns=["ticker", "score"]).to_csv(f, index=False)
    out = tmp_path / "esg.csv"
    monkeypatch.setattr(sys, "argv", ["build_esg.py", "--from-csv", str(f), "--higher-is-better", "--out", str(out), *extra])
    build_esg.main()
    return pd.read_csv(out)


def test_text_score_is_missing_not_ranked_as_best(monkeypatch, tmp_path, capsys):
    tk = UNIVERSES["extended"]
    rows = [(t, 20 + i) for i, t in enumerate(tk)]
    rows[2] = (tk[2], "Under Review")
    df = _run(monkeypatch, tmp_path, rows)
    assert tk[2] not in set(df["ticker"]) and len(df) == len(tk) - 1
    assert df["esg"].max() < 100 and "non-numeric" in capsys.readouterr().out


def test_too_many_missing_aborts(monkeypatch, tmp_path):
    tk = UNIVERSES["extended"]
    rows = [(t, 20 + i if i < 10 else None) for i, t in enumerate(tk)]
    with pytest.raises(SystemExit):
        _run(monkeypatch, tmp_path, rows)
