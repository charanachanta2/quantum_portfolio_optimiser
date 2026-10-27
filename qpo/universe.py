"""Ticker universes for the real-data run.

Every ticker below was listed and traded on a US exchange throughout 2004-2023 (so a 2008 backtest
can use it), BUT they are all *survivors*: Lehman Brothers, Bear Stearns, Washington Mutual,
Wachovia, Merrill Lynch, Countrywide, etc. are absent because they have no Yahoo history.  That is
survivorship bias and it flatters every method in the 2008 window (see report, Threats to validity).
`EXTENDED` adds names that were hammered in 2008 (but survived) to reduce - not remove - the bias.
Tickers whose Yahoo history is missing at the start date are dropped automatically and reported.
"""
CORE = ["AAPL", "MSFT", "JNJ", "PG", "KO", "PEP", "XOM", "CVX", "JPM", "WFC",
        "INTC", "CSCO", "MRK", "PFE", "WMT", "HD", "VZ", "T", "IBM", "ORCL"]
CRISIS_HIT = ["C", "BAC", "GS", "GE", "F", "AIG", "CAT", "DIS", "MCD", "BA"]
EXTENDED = CORE + CRISIS_HIT
UNIVERSES = {"core": CORE, "extended": EXTENDED}
