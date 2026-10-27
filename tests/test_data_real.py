"""Offline tests of the REAL-data code path using mocked Yahoo / FRED payloads
(the sandbox has no internet; these pin down the format handling and the no-look-ahead policy)."""
import io, os, warnings
import numpy as np, pandas as pd, pytest
from qpo.data import extract_close, parse_fred_csv, _finish, load_real_market


def _px(idx, tick):
    rng = np.random.default_rng(0)
    return pd.DataFrame(100 * np.cumprod(1 + rng.normal(0, 0.01, (len(idx), len(tick))), 0), index=idx, columns=tick)


IDX = pd.bdate_range("2005-01-03", "2007-12-31")
TICK = ["AAA", "BBB", "CCC"]


def test_extract_close_all_yfinance_layouts():
    px = _px(IDX, TICK)
    lay1 = pd.concat({"Close": px, "Open": px * 1.01}, axis=1)                  # (Price, Ticker)
    lay2 = pd.concat({t: pd.DataFrame({"Close": px[t], "Open": px[t]}) for t in TICK}, axis=1)   # (Ticker, Price)
    lay3 = pd.concat({"Adj Close": px, "Close": px * 0.9}, axis=1)              # old auto_adjust=False layout
    for lay in (lay1, lay2):
        assert np.allclose(extract_close(lay, TICK).values, px.values)
    assert np.allclose(extract_close(lay3, TICK).values, px.values)             # prefers Adj Close
    single = pd.DataFrame({"Close": px["AAA"], "Volume": 1.0})                  # flat single-ticker
    assert list(extract_close(single, ["AAA"]).columns) == ["AAA"]
    tz = lay1.copy(); tz.index = tz.index.tz_localize("America/New_York")
    assert extract_close(tz, TICK).index.tz is None                              # tz stripped


def test_fred_csv_new_and_old_headers_and_missing_dots():
    new = "observation_date,VIXCLS\n2008-01-02,23.17\n2008-01-03,.\n2008-01-04,23.94\n"
    old = "DATE,VIXCLS\n2008-01-02,23.17\n2008-01-03,.\n2008-01-04,23.94\n"
    for txt in (new, old):
        s = parse_fred_csv(pd.read_csv(io.StringIO(txt)), "VIXCLS")
        assert len(s) == 2 and s.iloc[0] == pytest.approx(23.17) and s.index[0] == pd.Timestamp("2008-01-02")


def test_finish_never_backfills_and_drops_truncated_macro():
    px = _px(IDX, TICK)
    vix = pd.Series(np.linspace(15, 30, len(IDX)), index=IDX, name="VIX")
    vix.iloc[:10] = np.nan                                                      # VIX starts late
    hy = pd.Series(5.0, index=IDX[-200:], name="HY_OAS")                        # truncated like FRED's ICE series
    macro = pd.concat([vix, hy], axis=1)
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        m = _finish(px, macro, pd.Series([50, 60, 70], index=TICK), "mock")
    assert "HY_OAS" not in m.macro.columns and "HY_OAS" in m.meta["dropped_macro"]
    assert m.macro["VIX"].notna().all()
    assert m.returns.index[0] >= IDX[10]                                        # rows without VIX were DROPPED, not back-filled
    assert m.macro["VIX"].iloc[0] == pytest.approx(vix.iloc[10])                # no future value copied backwards


def test_finish_drops_late_listed_ticker_and_reports():
    px = _px(IDX, TICK); px.loc[px.index < "2006-06-01", "CCC"] = np.nan
    vix = pd.Series(20.0, index=IDX, name="VIX")
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        m = _finish(px, vix.to_frame(), pd.Series([50, 60, 70], index=TICK), "mock")
    assert list(m.prices.columns) == ["AAA", "BBB"] and m.meta["dropped_tickers"] == ["CCC"]


def test_load_real_market_from_raw_csv_and_esg_guard(tmp_path):
    raw = tmp_path / "raw"; raw.mkdir()
    _px(IDX, TICK).to_csv(raw / "prices.csv")
    pd.DataFrame({"VIX": 20.0, "T10Y2Y": 1.0}, index=IDX).to_csv(raw / "macro.csv")
    with pytest.raises(FileNotFoundError):                                      # no silent pseudo-ESG
        load_real_market(TICK, "2005-01-01", "2007-12-31", None, str(raw))
    esg = tmp_path / "esg.csv"; pd.DataFrame({"ticker": TICK, "esg": [30, 55, 80]}).to_csv(esg, index=False)
    m = load_real_market(TICK, "2005-01-01", "2007-12-31", str(esg), str(raw))
    assert m.source == "yahoo+fred+esg" and m.esg["CCC"] == 80 and m.returns.shape[1] == 3
    m2 = load_real_market(TICK, "2005-01-01", "2007-12-31", None, str(raw), allow_pseudo_esg=True)
    assert "PSEUDO" in m2.source                                                # flagged in the source tag
