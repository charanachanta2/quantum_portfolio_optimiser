"""Market-data feeds for the live optimiser (A1).

`next_bar()` is the hook the statement's "streaming market feed" plugs into: a feed returns the next complete daily Bar, or None
when nothing new has arrived.  Two implementations:
  * ReplayFeed - replays historical daily bars (offline, tests, demos);
  * YahooFeed  - polls Yahoo Finance (via stock_prediction.data, TTL-cached) for bars newer than the last one seen.
Yahoo daily bars are end-of-day (intraday quotes are ~15 min delayed); the feed is therefore "live" at daily-bar granularity.
Intraday last-price quotes for the dashboard's mark-to-market line come from stock_prediction.data.fetch_quotes.
NOTE: YahooFeed has been tested against mocked payloads only (the build sandbox has no access to Yahoo).
"""
from __future__ import annotations
import time
from dataclasses import dataclass, replace
import numpy as np
import pandas as pd

from .data import Market


@dataclass
class Bar:
    date: pd.Timestamp
    returns: pd.Series          # simple daily returns indexed by ticker
    vix: float


def append_bar(m: Market, bar: Bar) -> Market:
    """New Market with one more trading day (prices extended by the returns, macro forward-filled except VIX)."""
    d = pd.Timestamp(bar.date)
    if d <= m.returns.index[-1]:
        raise ValueError(f"bar {d.date()} is not newer than the last bar {m.returns.index[-1].date()}")
    r = bar.returns.reindex(m.returns.columns).fillna(0.0)
    rets = pd.concat([m.returns, r.to_frame(d).T])
    px = pd.concat([m.prices, (m.prices.iloc[-1] * (1 + r)).to_frame(d).T])
    mac = m.macro.iloc[-1].copy(); mac["VIX"] = bar.vix
    macro = pd.concat([m.macro, mac.to_frame(d).T])
    return replace(m, returns=rets, prices=px, macro=macro)


def append_placeholder(m: Market, date) -> Market:
    """Adds a zero-return row dated `date` so a decision AT `date` can be built (make_instance(i = len-1)).
    The placeholder row is never read: instances use returns/macro strictly before index i."""
    return append_bar(m, Bar(pd.Timestamp(date), pd.Series(0.0, index=m.returns.columns), float(m.macro["VIX"].iloc[-1])))


class ReplayFeed:
    def __init__(self, returns: pd.DataFrame, vix: pd.Series, start: str | None = None):
        self.r, self.v = returns, vix
        self.k = int(np.searchsorted(returns.index, pd.Timestamp(start))) if start else 0

    def next_bar(self) -> Bar | None:
        if self.k >= len(self.r):
            return None
        b = Bar(self.r.index[self.k], self.r.iloc[self.k], float(self.v.iloc[self.k]))
        self.k += 1
        return b


class YahooFeed:
    def __init__(self, tickers, last_date, min_poll_seconds: float = 60.0, fetch=None):
        self.tickers = list(tickers)
        self.last_date = pd.Timestamp(last_date)
        self.min_poll = min_poll_seconds
        self._queue: list[Bar] = []
        self._t_last = 0.0
        if fetch is None:
            from stock_prediction import data as D
            fetch = D.fetch_prices
        self._fetch = fetch

    def bars_after(self, last_date=None) -> list[Bar]:
        last = pd.Timestamp(last_date or self.last_date)
        px = self._fetch(tuple(self.tickers) + ("^VIX",), start=str((last - pd.Timedelta(days=10)).date()))
        vix = px["^VIX"]
        px = px[self.tickers]
        px = px.loc[px.index <= px.index.max()]
        base = px.index[px.index <= last]
        if len(base) == 0:
            raise RuntimeError("downloaded window does not contain the last known bar; cannot chain returns")
        px = px.loc[base[-1]:]
        rets = px.pct_change().iloc[1:]
        rets = rets[rets.index > last]
        return [Bar(d, rets.loc[d], float(vix.reindex([d]).ffill().iloc[0]) if d in vix.index else float("nan")) for d in rets.index]

    def next_bar(self) -> Bar | None:
        if not self._queue and time.time() - self._t_last >= self.min_poll:
            self._t_last = time.time()
            self._queue = self.bars_after(self.last_date)
        if not self._queue:
            return None
        b = self._queue.pop(0)
        self.last_date = b.date
        return b


def catch_up(m: Market, feed: YahooFeed) -> tuple[Market, int]:
    """Append every daily bar published since the market's last date (e.g. snapshot -> today).  Returns (market, n_added)."""
    feed.last_date = m.returns.index[-1]
    bars = [b for b in feed.bars_after(feed.last_date) if not np.isnan(b.vix)]
    for b in bars:
        m = append_bar(m, b)
    feed.last_date = m.returns.index[-1]
    return m, len(bars)
