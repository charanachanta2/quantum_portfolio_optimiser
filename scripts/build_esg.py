"""Build data/esg.csv (columns: ticker, esg, raw_score, source, asof, method).  `esg` is in [0,100], HIGHER = BETTER.

Mode A (default) - Sustainalytics ESG *Risk* scores as shown on Yahoo Finance (via yfinance `Ticker.sustainability`):
    python scripts/build_esg.py --universe extended
    Sustainalytics scores are RISK scores (lower = better, typically 10-40 for large caps), so a raw
    "100 - risk" would put every large cap above the esg_min=40 hard filter and the filter would do nothing.
    We therefore convert to a cross-sectional percentile score:  esg = 100 * (1 - rank_pct(risk)).
    esg_min=40 then excludes roughly the 40 % highest-risk names of the universe.

Mode B - your own MSCI / Sustainalytics / other open subset:
    python scripts/build_esg.py --from-csv my_scores.csv --ticker-col Ticker --score-col "ESG Score" --higher-is-better
    (use --lower-is-better for risk-type scores).  MSCI letter ratings: map AAA..CCC to 7..1 beforehand.

Mode C - write an empty template to fill by hand:   python scripts/build_esg.py --template

Honest limitations (put them in the report): the scores are a *current snapshot* applied to 2004-2023
(look-ahead / static-ESG bias), rating agencies disagree strongly with each other, and Yahoo's
availability of this data has changed over time - if `Ticker.sustainability` returns nothing, use Mode B.
"""
import argparse, datetime as dt, os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from qpo.universe import UNIVERSES


def risk_to_score(raw: pd.Series, higher_is_better: bool) -> pd.Series:
    """Cross-sectional percentile score in [0,100], higher = better (ties -> average rank)."""
    pct = raw.rank(pct=True, method="average")
    pct = (pct - 1.0 / (2 * len(raw))) if len(raw) else pct          # centre ranks in (0,1)
    return (100 * (pct if higher_is_better else 1 - pct)).clip(0, 100)


def fetch_yahoo(tickers):
    import yfinance as yf
    rows, fails = [], 0
    for t in tickers:
        val = np.nan
        try:
            df = yf.Ticker(t).sustainability
            if df is not None and len(df):
                val = float(pd.to_numeric(df.loc["totalEsg"].iloc[0], errors="coerce"))
        except Exception as e:
            print(f"  {t}: no sustainability data ({type(e).__name__})")
        rows.append({"ticker": t, "raw_score": val})
        print(f"  {t}: {val}")
        fails = fails + 1 if not np.isfinite(val) else 0
        if fails >= 3 and not any(np.isfinite(r["raw_score"]) for r in rows):
            sys.exit("\nYahoo Finance no longer serves ESG/sustainability data (3 tickers in a row returned nothing) - "
                     "stopping instead of trying all of them.\nUse your own scores instead:\n"
                     "  python scripts/build_esg.py --template            # writes data/esg_template.csv (ticker list) to fill in\n"
                     "  python scripts/build_esg.py --from-csv my_scores.csv --ticker-col Ticker --score-col \"ESG Score\" --higher-is-better\n"
                     "(use --lower-is-better for risk-type scores).  For a smoke test only: add --allow-pseudo-esg to run_all.py.")
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--universe", default="extended", choices=list(UNIVERSES))
    ap.add_argument("--from-csv"); ap.add_argument("--ticker-col", default="ticker"); ap.add_argument("--score-col", default="score")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--higher-is-better", action="store_true"); g.add_argument("--lower-is-better", action="store_true")
    ap.add_argument("--source-name", default=None); ap.add_argument("--template", action="store_true")
    ap.add_argument("--out", default="data/esg.csv")
    ap.add_argument("--max-missing", type=int, default=3,
                    help="abort if more than this many tickers have no usable numeric score (missing ones get the median)")
    a = ap.parse_args()
    tickers = UNIVERSES[a.universe]
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    if a.template:
        pd.DataFrame({"ticker": tickers, "score": ""}).to_csv(a.out.replace(".csv", "_template.csv"), index=False)
        print("template written (columns: ticker,score)"); return
    # DEFAULT: if data/sp_esg.csv (S&P Global ESG Score, 0-100, higher = better) exists, use it instead of Yahoo.
    default_csv = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "sp_esg.csv")
    if not a.from_csv and os.path.exists(default_csv):
        a.from_csv = default_csv
        a.source_name = a.source_name or "S&P Global ESG Score (public lookup)"
        a.higher_is_better = True
        print(f"using {default_csv}")
    if a.from_csv:
        df = pd.read_csv(a.from_csv).rename(columns={a.ticker_col: "ticker", a.score_col: "raw_score"})
        df = df[df["ticker"].isin(tickers)][["ticker", "raw_score"]].copy()
        num = pd.to_numeric(df["raw_score"], errors="coerce")
        bad = df.loc[num.isna() & df["raw_score"].notna() & (df["raw_score"].astype(str).str.strip() != ""), ["ticker", "raw_score"]]
        if len(bad):                                   # e.g. "Under Review": text must never be ranked as if it were a number
            print("WARNING: non-numeric scores treated as MISSING:", dict(zip(bad["ticker"], bad["raw_score"])))
        df["raw_score"] = num
        hib = a.higher_is_better or not a.lower_is_better
        source = a.source_name or os.path.basename(a.from_csv); method = "percentile of " + ("score" if hib else "risk")
    else:
        df = fetch_yahoo(tickers); hib = False
        source = "Sustainalytics ESG Risk via Yahoo Finance (yfinance)"; method = "percentile of risk (lower risk -> higher score)"
    ok = df.dropna(subset=["raw_score"]).copy()
    missing = sorted(set(tickers) - set(ok["ticker"]))
    if len(missing) > a.max_missing:
        sys.exit(f"{len(missing)} of {len(tickers)} tickers have no numeric score (allowed: {a.max_missing}) - they would all get the median "
                 f"score and silently pass the esg_min filter. Fill them in or raise --max-missing. missing={missing}")
    ok["esg"] = risk_to_score(ok["raw_score"], hib).round(2).values
    ok["source"], ok["asof"], ok["method"] = source, dt.date.today().isoformat(), method
    ok[["ticker", "esg", "raw_score", "source", "asof", "method"]].to_csv(a.out, index=False)
    print(f"wrote {a.out}: {len(ok)} tickers; no score for: {missing or 'none'} (these get the median score + a warning)")


if __name__ == "__main__":
    main()
