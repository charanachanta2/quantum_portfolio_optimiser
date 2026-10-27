"""STAGE 0 (only step that needs internet):   python scripts/fetch_data.py [--universe core|extended]

Writes data/raw/prices.csv, data/raw/macro.csv and data/raw/provenance.json.
Everything downstream (`run_all.py --real`) then runs OFFLINE from these files, so the real-data
results are reproducible from a frozen snapshot.  Commit/zip data/raw together with results/.
"""
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from qpo.data import fetch_raw, RAW_DIR
from qpo.universe import UNIVERSES

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--universe", default="extended", choices=list(UNIVERSES))
    ap.add_argument("--start", default="2004-06-01")
    ap.add_argument("--end", default="2023-12-31")
    ap.add_argument("--raw-dir", default=RAW_DIR)
    a = ap.parse_args()
    prov = fetch_raw(UNIVERSES[a.universe], a.start, a.end, a.raw_dir)
    print(json.dumps({k: v for k, v in prov.items() if k != "price_missing_by_ticker"}, indent=1))
    miss = {k: v for k, v in prov["price_missing_by_ticker"].items() if v > 0}
    print("tickers with missing price rows (check before trusting):", miss or "none")
    print("NOTE: FRED's ICE-BofA series (HY_OAS) is limited to ~3 years since 2026-04 -> it is dropped by the loader.")
