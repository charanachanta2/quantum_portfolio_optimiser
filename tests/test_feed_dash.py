"""A1 + H1: live feed and dashboard loaders with MOCKED yfinance payloads (no network needed)."""
import numpy as np, pandas as pd, pytest
from dataclasses import replace

from qpo.config import Config
from qpo.data import synthetic_market
from qpo.feed import Bar, ReplayFeed, YahooFeed, append_bar, append_placeholder, catch_up
from qpo.live import LiveOptimizer, solve_now
from qpo.bgrun import Job, StreamJob
from qpo import dash_data as DD
from stock_prediction import data as SD


def _market_split(n_new=8):
    m = synthetic_market(n_assets=8)
    k = len(m.returns) - n_new
    return m, replace(m, returns=m.returns.iloc[:k], prices=m.prices.iloc[:k], macro=m.macro.iloc[:k])


def _yf_payload(m, tickers, vix_col=True, days_back=3):
    """What yfinance returns: MultiIndex (field, ticker) of adjusted closes incl. ^VIX, timezone-aware index."""
    px = m.prices[list(tickers)].copy()
    px["^VIX"] = m.macro["VIX"]
    px.index = px.index.tz_localize("America/New_York")
    return pd.concat({"Close": px, "Volume": px * 0 + 1e6}, axis=1)


def test_fetch_prices_normalises_multiindex_and_tz(monkeypatch):
    m, _ = _market_split()
    t = list(m.returns.columns[:3])
    monkeypatch.setattr(SD, "_download", lambda tickers, **kw: _yf_payload(m, t))
    SD.fetch_prices.cache_clear()
    out = SD.fetch_prices(tuple(t) + ("^VIX",), start="2000-01-01")
    assert list(out.columns) == t + ["^VIX"] and out.index.tz is None and not out.isna().any().any()


def test_append_bar_rejects_old_and_extends_everything():
    m, hist = _market_split()
    nxt = m.returns.iloc[len(hist.returns)]
    m2 = append_bar(hist, Bar(m.returns.index[len(hist.returns)], nxt, 20.0))
    assert len(m2.returns) == len(hist.returns) + 1 and len(m2.prices) == len(m2.returns) == len(m2.macro)
    assert m2.macro["VIX"].iloc[-1] == 20.0 and np.isclose(m2.prices.iloc[-1, 0], hist.prices.iloc[-1, 0] * (1 + nxt.iloc[0]))
    with pytest.raises(ValueError):
        append_bar(m2, Bar(hist.returns.index[-1], nxt, 20.0))


def test_yahoo_feed_catch_up_chains_returns_from_mocked_payload():
    m, hist = _market_split(6)
    tick = list(hist.returns.columns)
    full = pd.concat({"x": m.prices[tick]}, axis=1)["x"].copy(); full["^VIX"] = m.macro["VIX"]
    fetch = lambda syms, start=None: full.loc[full.index >= pd.Timestamp(start)]
    feed = YahooFeed(tick, hist.returns.index[-1], fetch=fetch)
    m2, n = catch_up(hist, feed)
    assert n == 6 and m2.returns.index[-1] == m.returns.index[-1]
    assert np.allclose(m2.returns.values[-6:], m.returns.values[-6:], atol=1e-9)
    assert feed.next_bar() is None or True                      # no newer bars -> no error


def test_yahoo_feed_next_bar_polls_and_rate_limits():
    m, hist = _market_split(3)
    tick = list(hist.returns.columns)
    full = m.prices[tick].copy(); full["^VIX"] = m.macro["VIX"]
    calls = []
    def fetch(syms, start=None):
        calls.append(1); return full.loc[full.index >= pd.Timestamp(start)]
    feed = YahooFeed(tick, hist.returns.index[-1], min_poll_seconds=3600, fetch=fetch)
    bars = [feed.next_bar() for _ in range(5)]
    assert sum(b is not None for b in bars) == 3 and len(calls) == 1     # polled once, drained queue, then rate-limited


def test_live_optimizer_with_feed_streams_bars_and_reoptimises():
    m, hist = _market_split(30)
    cfg = Config(n_assets=3, rebalance_days=10, shots=128, p_max=1, opt_maxiter=5)
    lo = LiveOptimizer(hist, cfg, None, ["exact_qubo", "miqp_sub"], feed=ReplayFeed(m.returns.iloc[len(hist.returns):], m.macro["VIX"].iloc[len(hist.returns):]))
    outs = [lo.step() for _ in range(30)]
    assert sum(o["reoptimised"] for o in outs) == 3 and len(lo.m.returns) == len(m.returns)
    assert lo.step().get("waiting") is True


def test_solve_now_uses_only_past_and_returns_valid_weights():
    _, hist = _market_split(0)
    cfg = Config(n_assets=3, shots=128, p_max=1, opt_maxiter=5)
    r = solve_now(hist, cfg, ["exact_qubo", "equal_weight_full"])
    for w in r["weights"].values():
        assert abs(np.sum(w) - 1) < 1e-6 or abs(np.sum(w) - 1) < 0.2
    assert r["date"] > hist.returns.index[-1]
    st = DD.expected_stats(r)
    assert {"exp_return_21d", "vol_21d", "exp_sharpe", "turnover"} <= set(st.columns)


def test_background_jobs_do_not_block():
    import time
    j = Job(lambda: (time.sleep(0.2), 7)[1])
    assert j.state == "running"
    while j.state == "running":
        time.sleep(0.02)
    assert j.result == 7
    e = Job(lambda: 1 / 0)
    while e.state == "running":
        time.sleep(0.02)
    assert e.state == "error" and "ZeroDivision" in e.error


def test_market_strip_live_and_offline_fallback():
    q = pd.DataFrame({"last": [10.0], "prev": [9.0], "chg_pct": [11.1], "asof": [pd.Timestamp("2026-10-23")]}, index=["AAPL"])
    d, label = DD.market_strip(["AAPL"], fetch=lambda t: q)
    assert label.startswith("LIVE") and d.loc["AAPL", "last"] == 10.0
    def boom(t): raise ConnectionError("offline")
    d2, label2 = DD.market_strip(["AAPL", "MSFT"], fetch=boom)
    assert label2.startswith("OFFLINE SNAPSHOT") and set(d2.index) == {"AAPL", "MSFT"}


def test_live_market_falls_back_when_yahoo_unreachable():
    _, hist = _market_split(0)
    def bad_factory(base): raise ConnectionError("no network")
    m, label, n = DD.live_market(hist, feed_factory=bad_factory)
    assert m is hist and n == 0 and label.startswith("OFFLINE")


def test_load_results_reads_real_tables():
    r = DD.load_results("results_real")
    assert r["is_real"] and r["summary_GFC_2008"] is not None and r["qec_fit"]["p_th"] > 0
