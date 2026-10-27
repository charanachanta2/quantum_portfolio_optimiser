"""Gemini advisor from the command line: solve once, rank the stocks, write a summary (what we did / how / what happens next).

    set GEMINI_API_KEY=...            (PowerShell: $env:GEMINI_API_KEY="...";  bash: export GEMINI_API_KEY=...)
    python scripts/advise.py --check-gemini            # one tiny call: proves key + model + endpoint work
    python scripts/advise.py                           # offline snapshot (2004-2023), results from results_real/
    python scripts/advise.py --live                    # extend the snapshot with Yahoo bars since 2023-12-29 (internet)
    python scripts/advise.py --no-llm                  # deterministic template, no network call at all
    python scripts/advise.py --prompt-key              # type the key instead of using an environment variable

Writes advice/advice_<date>.md and .json.  The pick and all numbers come from the optimiser code; Gemini only writes the narrative.
Without a key (or if Gemini fails) the same structure is produced by an offline template, so this never blocks a demo.
Research demo - not financial advice."""
import argparse, getpass, json, os, sys, warnings
from datetime import date
from dataclasses import replace
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qpo import advisor
from qpo.config import Config

DEFAULT_METHODS = ["qaoa_noisy_mitigated", "qaoa_logical", "exact_qubo", "sa_qubo", "miqp_sub", "cvar_sub", "equal_weight_full"]


def check_gemini(key, model):
    text, used = advisor.gemini_generate("Reply with the JSON object {\"ok\": true} and nothing else.", "ping", api_key=key, model=model, max_retries=2, timeout=30)
    print(f"Gemini OK - model {used} answered: {text.strip()[:80]}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default="results_real"); ap.add_argument("--out", default="advice")
    ap.add_argument("--live", action="store_true"); ap.add_argument("--no-llm", action="store_true")
    ap.add_argument("--model", default=None, help=f"default {advisor.DEFAULT_MODEL} (or env GEMINI_MODEL)")
    ap.add_argument("--prompt-key", action="store_true"); ap.add_argument("--check-gemini", action="store_true")
    ap.add_argument("--n-assets", type=int, default=3, choices=[3, 4]); ap.add_argument("--shots", type=int, default=512)
    ap.add_argument("--mu-source", default="historical", choices=["historical", "lstm", "blend"])
    ap.add_argument("--esg-min", type=float, default=40.0); ap.add_argument("--methods", nargs="+", default=DEFAULT_METHODS)
    a = ap.parse_args()

    key = getpass.getpass("Gemini API key (hidden): ").strip() if a.prompt_key else advisor.resolve_api_key()
    model = a.model or os.environ.get("GEMINI_MODEL") or advisor.DEFAULT_MODEL
    if a.check_gemini:
        if not key:
            sys.exit("No key: set GEMINI_API_KEY (or use --prompt-key).")
        try:
            check_gemini(key, model)
        except advisor.GeminiError as e:
            sys.exit(advisor._scrub(str(e), key))
        return

    from qpo.data import load_real_market
    from qpo.universe import UNIVERSES
    from qpo.live import solve_now
    from qpo import dash_data as DD, qec
    market = load_real_market(UNIVERSES["extended"], "2004-06-01", "2023-12-31", "data/esg.csv", "data/raw")
    label = f"offline snapshot, last bar {market.returns.index[-1].date()}"
    if a.live:
        market, label, _ = DD.live_market(market)
    print("Data:", label)
    cfg = Config(n_assets=a.n_assets, shots=a.shots, p_max=2, opt_maxiter=30, esg_min=a.esg_min, mu_source=a.mu_source)
    fc = None
    if a.mu_source != "historical":
        from stock_prediction.model import Predictor
        from stock_prediction.forecast import LSTMForecaster
        fc = LSTMForecaster(Predictor.load("models", "deployed"), market.prices)
    results = DD.load_results(a.results)
    if not results["meta"]:
        print(f"WARNING: no results found in {a.results}/ - the fault-tolerance and backtest evidence will be empty.")
    fit = results.get("qec_fit")
    lq = qec.logical_q_per_layer(fit, cfg.logical_distance, cfg.logical_p_phys) if fit else 1e-6
    print("Solving ...", ", ".join(a.methods))
    res = solve_now(market, cfg, a.methods, logical_q=lq, forecaster=fc)
    adv = advisor.advise(res, results, cfg, api_key=key, model=model, use_llm=not a.no_llm, esg=market.esg)
    os.makedirs(a.out, exist_ok=True)
    stem = os.path.join(a.out, f"advice_{date.today().isoformat()}")
    open(stem + ".md", "w", encoding="utf-8").write(advisor.to_markdown(adv))
    json.dump(adv, open(stem + ".json", "w", encoding="utf-8"), indent=1, default=str)
    p = adv["pick"]
    print(f"\nTop consensus pick: {p['ticker']}  (consensus weight {p['consensus_weight']:.1%}, confidence {p['confidence']})")
    print(f"Narrative source: {adv['source']}" + (f" ({adv['model']})" if adv["model"] else "") + (f"  [Gemini failed: {adv['llm_error']}]" if adv["llm_error"] else ""))
    w = adv["warnings"]
    if w["unverified_numbers"] or w["unexpected_tickers"]:
        print("Unverified items in the narrative:", w)
    print(f"Wrote {stem}.md and {stem}.json")


if __name__ == "__main__":
    main()
