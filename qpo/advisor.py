"""Gemini advisor: turns one optimiser solve (+ the fault-tolerance / backtest results) into a stock pick and a written summary.

Design rule: **the pick and every number are computed by this code, never by the LLM.**
  1. `build_evidence`   - collects facts from `solve_now` output and `results_real/` and ranks the candidate stocks deterministically
                          (consensus weight across the optimisers; tie-break by model-implied return/vol).
  2. Gemini (REST)      - receives ONLY that evidence as JSON and writes the narrative (what we did, how, what happens next).
                          Its JSON is validated; numbers / tickers that are not in the evidence are flagged; one corrective retry.
  3. `template_narrative` - deterministic offline fallback with the same structure (no key, no internet, or Gemini failed).
The honesty notes (data staleness, 0 significant quantum-vs-classical differences, simulated quantum hardware, ...) are attached by
code in `fixed_notes`, so the LLM cannot drop them.  Research demo - not financial advice.

Only the standard library is used for the HTTP call (no extra dependency).  The API key is read from the environment
(GEMINI_API_KEY or GOOGLE_API_KEY) or passed in; it is sent in the `x-goog-api-key` header and scrubbed from error messages."""
from __future__ import annotations
import json
import math
import os
import re
import time
import urllib.error
import urllib.request
from datetime import date, datetime, timezone

import numpy as np

DEFAULT_MODEL = "gemini-3.8-flash"          # GA Flash model per the Gemini changelog (Oct 2026); override with GEMINI_MODEL / --model
FALLBACK_MODELS = ("gemini-3.6-flash",)     # tried if the requested model returns 404
ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
BASELINES = {"equal_weight_full"}           # reference portfolio, not an optimiser: excluded from the consensus
DISCLAIMER = ("Research demo - not financial advice. Quantum solvers run on a classical simulator and logical-qubit noise is modelled. "
              "The ranking reflects a mean-variance QUBO on historical data, not a forecast of future returns.")
STALE_DAYS = 10
SECTIONS = ("what_we_did", "how_it_works", "fault_tolerance_findings", "honest_caveats", "what_happens_next")


# ----------------------------------------------------------------------------- small helpers
def _f(x, nd=4):
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return round(x, nd) if math.isfinite(x) else None


def _pct(x, nd=1):
    return f"{100 * x:.{nd}f}%"


def resolve_api_key(explicit: str | None = None) -> str | None:
    return (explicit or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY") or "").strip() or None


def _scrub(msg: str, key: str | None) -> str:
    return msg.replace(key, "***") if key else msg


# ----------------------------------------------------------------------------- 1. evidence (all numbers computed here)
def rank_candidates(res: dict, esg=None) -> tuple[list[dict], dict]:
    """Rank tickers by consensus weight over the optimiser methods in `res` (equal-weight baseline excluded)."""
    tickers = list(res["tickers"])
    methods = [k for k in res["weights"] if k not in BASELINES] or list(res["weights"])
    quantum = [k for k in methods if k.startswith("qaoa_")]
    classical = [k for k in methods if k not in quantum]
    elig = list(res.get("elig", []))
    mu = dict(zip(elig, np.asarray(res.get("mu_elig", []), float)))
    sd = dict(zip(elig, np.sqrt(np.clip(np.diag(np.asarray(res["Sigma_elig"], float)), 1e-18, None)))) if "Sigma_elig" in res else {}
    W = {k: np.asarray(res["weights"][k], float) for k in methods}
    top = {k: tickers[int(np.argmax(W[k]))] for k in methods}
    rows = []
    for i, t in enumerate(tickers):
        ws = [W[k][i] for k in methods]
        if max(ws) <= 1e-9:
            continue
        qm = float(np.mean([W[k][i] for k in quantum])) if quantum else None
        cm = float(np.mean([W[k][i] for k in classical])) if classical else None
        r = mu.get(t)
        v = sd.get(t)
        rows.append({"ticker": t, "consensus_weight": _f(np.mean(ws)), "methods_holding": int(sum(w > 1e-9 for w in ws)),
                     "quantum_mean_weight": _f(qm), "classical_mean_weight": _f(cm), "top_holding_of": [k for k in methods if top[k] == t],
                     "exp_return_21d": _f(r), "vol_21d": _f(v), "return_per_vol": _f(r / v) if (r is not None and v) else None,
                     "esg_percentile": _f(esg[t], 1) if esg is not None and t in esg else None})
    rows.sort(key=lambda d: (-(d["consensus_weight"] or 0), -(d["return_per_vol"] if d["return_per_vol"] is not None else -1e9), d["ticker"]))
    info = {"methods": methods, "quantum_methods": quantum, "classical_methods": classical, "top_by_method": top}
    return rows, info


def _argmax_mean(res, methods):
    if not methods:
        return None
    m = np.mean([np.asarray(res["weights"][k], float) for k in methods], axis=0)
    return list(res["tickers"])[int(np.argmax(m))]


def _backtest_evidence(results: dict) -> dict:
    out = {}
    for p in ("GFC_2008", "COVID_2020"):
        s, t = results.get(f"summary_{p}"), results.get(f"signif_{p}")
        if s is None:
            continue
        sh = s["sharpe"].astype(float)
        q = sh[[i for i in sh.index if i.startswith("qaoa_")]]
        c = sh[[i for i in sh.index if not i.startswith("qaoa_")]]
        d = {"top3_by_sharpe": {k: _f(v, 3) for k, v in sh.sort_values(ascending=False).head(3).items()},
             "best_quantum": {"method": q.idxmax(), "sharpe": _f(q.max(), 3)} if len(q) else None,
             "best_classical": {"method": c.idxmax(), "sharpe": _f(c.max(), 3)} if len(c) else None,
             "equal_weight_sharpe": _f(sh.get("equal_weight_full"), 3)}
        if t is not None and "p_boot_holm" in t:
            d["significant_diffs"] = int((t["p_boot_holm"] < 0.05).sum())
            d["total_comparisons"] = int(len(t))
        out[p] = d
    return out


def _ft_evidence(res: dict, results: dict, cfg) -> dict:
    ft: dict = {}
    if cfg is not None:
        try:
            from .dash_data import quantum_internals
            p = int(next((v.get("p") for v in res["info"].values() if v.get("p")), cfg.p_max))
            qi = quantum_internals(res, cfg, p, results)
            qi.pop("resource_row", None)
            ft["circuit"] = {k: (_f(v, 6) if isinstance(v, float) else v) for k, v in qi.items()}
        except Exception as e:  # evidence is best-effort; the pick never depends on it
            ft["circuit_error"] = type(e).__name__
    rc = results.get("resource_corners")
    if rc is not None and len(rc):
        base = rc[(rc["scenario"] == "baseline")]
        ft["resource_estimates_baseline"] = [{"circuit": r["circuit"], "p_phys": _f(r["p_phys"], 6), "distance": int(r["distance"]),
                                              "physical_qubits": int(r["physical_qubits"]), "time_per_shot_ms": _f(r["time_per_shot_ms"], 2)}
                                             for _, r in base.iterrows() if not math.isnan(float(r["distance"]))]
    ab = results.get("ablation")
    if ab is not None:
        ft["mitigation_ablation"] = {k: {"est_err": _f(r["est_err"]), "overhead": _f(r["overhead"], 2), "q_eff": _f(r["q_eff"])} for k, r in ab.iterrows()}
    fit = results.get("qec_fit")
    if fit:
        ft["stim_fit"] = {"p_th": _f(fit.get("p_th"), 5), "n_points": fit.get("n_points")}
    sf = results.get("scaling_fits")
    if sf is not None:
        ft["scaling_log2_slope_per_qubit"] = {k: _f(v, 3) for k, v in sf.iloc[:, 0].items()}
    return ft


def build_evidence(res: dict, results: dict | None = None, cfg=None, today: date | None = None, esg=None) -> dict:
    """Everything the narrative may talk about.  `res` = qpo.live.solve_now output, `results` = qpo.dash_data.load_results('results_real')."""
    results = results or {}
    today = today or date.today()
    cands, info = rank_candidates(res, esg)
    if not cands:
        raise ValueError("no stock has a positive weight in any method")
    pick = dict(cands[0])
    asof = res["date"].date() if hasattr(res["date"], "date") else res["date"]
    stale_days = (today - asof).days
    n_m = len(info["methods"])
    agree = sum(1 for k in info["methods"] if info["top_by_method"][k] == pick["ticker"]) / n_m
    q_top, c_top = _argmax_mean(res, info["quantum_methods"]), _argmax_mean(res, info["classical_methods"])
    conf = "high" if (agree >= 0.75 and pick["methods_holding"] == n_m) else "medium" if agree >= 0.5 else "low"
    stale = stale_days > STALE_DAYS
    if stale:
        conf = "low"
    pick.update({"agreement_share": _f(agree, 2), "quantum_top": q_top, "classical_top": c_top,
                 "quantum_classical_agree": (q_top == c_top) if (q_top and c_top) else None, "confidence": conf})
    meta = results.get("meta", {}) or {}
    dm = meta.get("data_meta", {}) or {}
    bt = _backtest_evidence(results)
    sig = [(d["significant_diffs"], d["total_comparisons"]) for d in bt.values() if "significant_diffs" in d]
    ev = {
        "run": {"decision_date": str(asof), "today": str(today), "days_since_decision_date": stale_days, "stale": stale,
                "universe_size": len(res["tickers"]), "eligible_after_esg_filter": len(res.get("elig", [])),
                "esg_min": _f(cfg.esg_min, 1) if cfg is not None else None,
                "assets_on_qubits": res.get("sub"), "qubits": res.get("n_qubits"), "mu_source": res.get("mu_source"),
                "risk_source": res.get("risk_source"), "lambda_t": [_f(x, 3) for x in res.get("lam", [])], "weight_cap_levels_of_6": res.get("caps"),
                "tc_bps": _f(cfg.tc_bps, 1) if cfg is not None else None, "shots": cfg.shots if cfg is not None else None,
                "methods_in_consensus": info["methods"], "quantum_methods": info["quantum_methods"], "classical_methods": info["classical_methods"]},
        "pick": pick, "runner_ups": cands[1:4], "candidates_all": [c["ticker"] for c in cands],
        "weights_by_method": {k: {t: _f(w) for t, w in zip(res["tickers"], res["weights"][k]) if w > 1e-6} for k in res["weights"]},
        "solve_quality": {k: {"approx_ratio": _f(v.get("approx_ratio")), "is_opt": v.get("is_opt"), "turnover": _f(v.get("turnover"), 3)} for k, v in res["info"].items()},
        "fault_tolerance": _ft_evidence(res, results, cfg),
        "backtests": bt,
        "provenance": {"backtest_data_source": meta.get("data_source"), "backtests_are_real": bool(results.get("is_real")),
                       "esg_source": dm.get("esg_provenance"), "esg_missing_tickers": dm.get("esg_missing", []), "dropped_macro": dm.get("dropped_macro", []),
                       "history": [meta.get("first_date"), meta.get("last_date")]},
    }
    notes = [DISCLAIMER]
    if stale:
        notes.append(f"STALE DATA: the solve uses market data up to {asof} ({stale_days} days before today). This is a demonstration on a snapshot, not a current pick; "
                     "re-run with the live feed before using it.")
    if results and not results.get("is_real"):
        notes.append("The backtest results folder is not marked as real data (SYNTHETIC or missing): treat its numbers as plumbing checks only.")
    if sig:
        s, n = sum(a for a, _ in sig), sum(b for _, b in sig)
        notes.append(f"Across the crisis backtests ({len(sig)} windows) {s} of {n} quantum-vs-classical Sharpe differences are statistically significant (Holm-corrected): "
                     "no quantum advantage is demonstrated, and the pick should not be read as one.")
    if dm.get("esg_missing"):
        notes.append(f"No ESG score for {', '.join(dm['esg_missing'])} (filled with the median).")
    ev["fixed_notes"] = notes
    return ev


# ----------------------------------------------------------------------------- 3. deterministic fallback narrative
def template_narrative(ev: dict) -> dict:
    r, p, ft, bt = ev["run"], ev["pick"], ev["fault_tolerance"], ev["backtests"]
    cand_txt = ", ".join(f"{c['ticker']} ({_pct(c['consensus_weight'])})" for c in ev["runner_ups"]) or "none"
    circ = ft.get("circuit", {})
    res_rows = ft.get("resource_estimates_baseline", [])
    ab = ft.get("mitigation_ablation", {})
    what = [f"Loaded {r['universe_size']} stocks and kept {r['eligible_after_esg_filter']} after the hard ESG filter (minimum percentile {r['esg_min']}).",
            f"Built a time-dependent QUBO on {r['qubits']} qubits for the assets {', '.join(r['assets_on_qubits'] or [])}, with VIX-driven risk aversion {r['lambda_t']}, "
            f"weight caps of {r['weight_cap_levels_of_6']} units out of 6, transaction costs of {r['tc_bps']} bps and an ESG tilt.",
            f"Solved it with {len(r['methods_in_consensus'])} optimisers: quantum ({', '.join(r['quantum_methods']) or 'none'}) and classical ({', '.join(r['classical_methods'])}).",
            f"Ranked every stock by its average weight across those optimisers. Top of the ranking: {p['ticker']} ({_pct(p['consensus_weight'])} average weight; "
            f"held by {p['methods_holding']} of {len(r['methods_in_consensus'])} methods). Next: {cand_txt}."]
    how = ["QAOA runs on an exact statevector simulator with an adaptive number of layers; the classical solvers are exact search, simulated annealing, MIQP and a CVaR scenario program.",
           "Noise is modelled as an effective bit-flip probability per qubit derived from gate counts; dynamical decoupling, zero-noise extrapolation and probabilistic error cancellation are applied on top.",
           "The logical-qubit variant takes its error rate from Stim/PyMatching surface-code memory simulations fitted to a threshold model; the QAOA circuit itself sees that rate as an effective flip probability.",
           f"Decision date {r['decision_date']}; expected returns come from '{r['mu_source']}' and the risk-aversion driver is '{r['risk_source']}'."]
    ftf = []
    if circ.get("q_eff_raw") is not None:
        s = f"Effective bit-flip probability per qubit: {circ['q_eff_raw']:.4f} raw, {circ['q_eff_dd']:.4f} with dynamical decoupling"
        if circ.get("q_eff_logical") is not None:
            s += f", {circ['q_eff_logical']:.2e} on modelled logical qubits (distance {circ['distance']}, physical error {circ['p_phys_logical']:g})"
        ftf.append(s + ".")
    if circ.get("p_th_fit"):
        ftf.append(f"The Stim fit gives a surface-code threshold of about {circ['p_th_fit']:.4f}.")
    for row in res_rows[:2]:
        ftf.append(f"Baseline resource estimate, {row['circuit']} at physical error {row['p_phys']:g}: distance {row['distance']}, {row['physical_qubits']} physical qubits, {row['time_per_shot_ms']} ms per shot.")
    if "raw" in ab and "ZNE" in ab and "PEC" in ab:
        ftf.append(f"Energy-estimation error: {ab['raw']['est_err']} raw, {ab['ZNE']['est_err']} with ZNE (overhead {ab['ZNE']['overhead']}x), {ab['PEC']['est_err']} with PEC (overhead {ab['PEC']['overhead']}x).")
    sl = ft.get("scaling_log2_slope_per_qubit", {})
    if "N99_qaoa" in sl and "SA_evals_median" in sl:
        ftf.append(f"Shots-to-optimum scale as 2^({sl['N99_qaoa']} n) for QAOA against 2^({sl['SA_evals_median']} n) for simulated annealing (n = 6..16 qubits only): no scaling advantage.")
    ftf = ftf or ["No fault-tolerance result files were available for this run."]
    cav = [f"Only {len(r['assets_on_qubits'] or [])} of the {r['eligible_after_esg_filter']} ESG-eligible stocks are encoded on qubits (pre-selected greedily); the other stocks can only appear through the full-universe baselines.",
           "The ESG scores are a static snapshot, the universe contains only survivors, and the quantum noise is a phenomenological model: all three can flatter the results."]
    for pn, d in bt.items():
        if d.get("best_quantum") and d.get("best_classical"):
            cav.append(f"{pn}: best quantum Sharpe {d['best_quantum']['sharpe']} ({d['best_quantum']['method']}) vs best classical {d['best_classical']['sharpe']} ({d['best_classical']['method']}).")
    nxt = ["The target weights become the portfolio to rebalance towards; the live dashboard re-solves every 21 trading days and this advisor can be re-run after each solve.",
           f"Before acting on {p['ticker']}, refresh the data (live feed) and check the stock against information this model does not see: fundamentals, news, liquidity.",
           "Open items: a longer scaling study with confidence intervals, VQE as an alternative ansatz, and a larger investable universe encoded on qubits."]
    return {"headline": f"{p['ticker']} has the highest consensus weight ({_pct(p['consensus_weight'])}) across {len(r['methods_in_consensus'])} optimisers; confidence {p['confidence']}.",
            "what_we_did": what, "how_it_works": how, "fault_tolerance_findings": ftf, "honest_caveats": cav, "what_happens_next": nxt}


# ----------------------------------------------------------------------------- 2. Gemini REST client
class GeminiError(RuntimeError):
    def __init__(self, status: int | None, message: str):
        super().__init__(f"Gemini API error ({status}): {message}")
        self.status = status


_urlopen = urllib.request.urlopen     # patched in tests
_sleep = time.sleep


def gemini_generate(system: str, user: str, *, api_key: str, model: str = DEFAULT_MODEL, timeout: float = 90, max_retries: int = 3,
                    temperature: float = 0.2) -> tuple[str, str]:
    """POST generateContent; returns (text, model_used).  Retries 429/5xx with backoff; tries FALLBACK_MODELS on 404."""
    body = json.dumps({"systemInstruction": {"parts": [{"text": system}]},
                       "contents": [{"role": "user", "parts": [{"text": user}]}],
                       "generationConfig": {"temperature": temperature, "responseMimeType": "application/json", "maxOutputTokens": 8192}}).encode()
    last: Exception | None = None
    for m in (model,) + tuple(x for x in FALLBACK_MODELS if x != model):
        for attempt in range(max_retries):
            req = urllib.request.Request(ENDPOINT.format(model=m), data=body, method="POST",
                                         headers={"Content-Type": "application/json", "x-goog-api-key": api_key})
            try:
                with _urlopen(req, timeout=timeout) as resp:
                    data = json.loads(resp.read().decode())
            except urllib.error.HTTPError as e:
                try:
                    msg = json.loads(e.read().decode()).get("error", {}).get("message", "")
                except Exception:
                    msg = ""
                last = GeminiError(e.code, _scrub(msg or e.reason or "HTTP error", api_key))
                if e.code == 404:
                    break                                   # model not found -> next model
                if e.code in (429, 500, 502, 503, 504) and attempt < max_retries - 1:
                    _sleep(2 ** attempt); continue
                raise last
            except (urllib.error.URLError, TimeoutError, OSError) as e:
                last = GeminiError(None, _scrub(str(getattr(e, "reason", e)), api_key))
                if attempt < max_retries - 1:
                    _sleep(2 ** attempt); continue
                raise last
            cands = data.get("candidates") or []
            if not cands:
                raise GeminiError(None, f"no candidates returned (promptFeedback={data.get('promptFeedback')})")
            text = "".join(p.get("text", "") for p in (cands[0].get("content") or {}).get("parts", []))
            if not text.strip():
                raise GeminiError(None, f"empty response (finishReason={cands[0].get('finishReason')})")
            return text, m
    raise last or GeminiError(None, "no model could be reached")


SYSTEM_PROMPT = """You are the narrator of a portfolio-optimisation research demo (QAOA / quantum error mitigation / fault-tolerance study).
You receive EVIDENCE as JSON. Everything in it was computed by code. Write a summary of what the pipeline did, how, what the fault-tolerance results show, and what happens next.

HARD RULES
- The stock PICK is fixed by the code (evidence.pick). Explain it; never choose, add or swap stocks.
- Use only numbers that appear in EVIDENCE, copied with the same digits. Never invent, estimate, round differently or compute new numbers.
- Never claim a quantum advantage. If significant_diffs is 0, say plainly that no advantage was found. Quantum hardware is simulated; logical-qubit noise is modelled.
- If run.stale is true, say prominently that the data is an old snapshot and the pick is a demonstration, not a current recommendation.
- Not financial advice; do not use words like "buy" or "guaranteed".
- Plain, concrete language for a technically literate reader. Each bullet at most 2 sentences. Whole answer under 450 words.

OUTPUT: a single JSON object, no markdown fences, with exactly these keys:
{"headline": string,
 "what_we_did": [3-5 strings],
 "how_it_works": [3-4 strings],
 "fault_tolerance_findings": [3-5 strings],
 "honest_caveats": [3-5 strings],
 "what_happens_next": [3-4 strings]}"""


def _parse_json(text: str) -> dict:
    t = text.strip()
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", t)
    a, b = t.find("{"), t.rfind("}")
    if a < 0 or b < a:
        raise ValueError("no JSON object in the response")
    return json.loads(t[a:b + 1])


def validate_narrative(d: dict) -> dict:
    if not isinstance(d, dict) or not isinstance(d.get("headline"), str) or not d["headline"].strip():
        raise ValueError("missing 'headline'")
    out = {"headline": d["headline"].strip()[:400]}
    for k in SECTIONS:
        v = d.get(k)
        if isinstance(v, str):
            v = [v]
        if not isinstance(v, list) or not v or not all(isinstance(x, str) and x.strip() for x in v):
            raise ValueError(f"'{k}' must be a non-empty list of strings")
        out[k] = [x.strip()[:600] for x in v[:8]]
    return out


def _evidence_numbers(ev) -> list[float]:
    nums: list[float] = []

    def walk(x):
        if isinstance(x, bool) or x is None:
            return
        if isinstance(x, (int, float)):
            if math.isfinite(x):
                nums.append(float(x))
        elif isinstance(x, str):
            nums.extend(float(m) for m in re.findall(r"\d+(?:\.\d+)?", x.replace(",", "")))
        elif isinstance(x, dict):
            for k, v in x.items():
                walk(str(k)); walk(v)
        elif isinstance(x, (list, tuple)):
            for v in x:
                walk(v)
    walk(ev)
    return nums


def check_narrative(narr: dict, ev: dict) -> dict:
    """Flag numbers / tickers in the narrative that the evidence does not support (warnings, not proof of error)."""
    text = " ".join([narr["headline"]] + [s for k in SECTIONS for s in narr[k]])
    text_n = re.sub(r"(?<=\d),(?=\d{3})", "", text)
    pool = _evidence_numbers(ev)
    bad = []
    # the evidence pool holds unsigned values from text and signed values from numeric leaves; a '-' written in the narrative must match a negative leaf
    signed = [e for e in pool]
    for m in re.finditer(r"(?<![\w.])(-?)(\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)(\s?%)?", text_n):
        neg, s, pct = bool(m.group(1)), m.group(2), bool(m.group(3))
        v = float(s)
        if re.fullmatch(r"\d+", s) and not pct and (v <= 12 or 1990 <= v <= 2100):
            continue                                        # small counts, years
        mant = re.split(r"[eE]", s)[0]
        dec = len(mant.split(".")[1]) if "." in mant else 0
        cands = [v / 100.0] if pct else [v, v / 100.0]
        if neg:
            cands = [-c for c in cands]
        tol = 0.5 * 10 ** (-dec) * (0.01 if pct else 1)
        close = lambda c, e, t: abs(c - e) <= max(t * 1.01, 0.005 * abs(e))
        ok = any(close(c, e, tol) for c in cands for e in signed)
        if not ok and not neg:                              # unsigned text may legitimately match a negative leaf ("a drawdown of 0.33")
            ok = any(close(c, abs(e), tol) for c in cands for e in signed)
        if not ok and pct:
            ok = any(close(-v if neg else v, e, 0.5 * 10 ** (-dec)) for e in signed)
        if not ok:
            bad.append(m.group(0).strip())
    allowed = set(ev["candidates_all"]) | set(ev["provenance"].get("esg_missing_tickers") or []) | set(ev["run"].get("assets_on_qubits") or [])
    universe = set(ev["candidates_all"]) | allowed
    universe |= set(ev.get("universe_tickers", []))
    mentioned = {t for t in universe if len(t) >= 2 and re.search(rf"\b{re.escape(t)}\b", text)}
    return {"unverified_numbers": sorted(set(bad)), "unexpected_tickers": sorted(mentioned - allowed)}


def _llm_narrative(ev: dict, key: str | None, model: str | None, generate) -> tuple[dict, str, dict]:
    user = "EVIDENCE:\n" + json.dumps({k: v for k, v in ev.items() if k != "universe_tickers"}, indent=1, default=str)
    feedback, narr, used, issues = "", None, model or DEFAULT_MODEL, {"unverified_numbers": [], "unexpected_tickers": []}
    for attempt in range(2):
        prompt = user + feedback
        text, used = generate(SYSTEM_PROMPT, prompt) if generate else gemini_generate(SYSTEM_PROMPT, prompt, api_key=key, model=model or DEFAULT_MODEL)
        try:
            cand = validate_narrative(_parse_json(text))
        except (ValueError, json.JSONDecodeError) as e:
            feedback = f"\n\nYour previous answer was rejected: {e}. Return only the JSON object described in the instructions."
            if attempt == 1:
                raise ValueError(f"Gemini returned invalid JSON twice: {e}")
            continue
        narr, issues = cand, check_narrative(cand, ev)
        if not issues["unverified_numbers"] and not issues["unexpected_tickers"]:
            break
        feedback = ("\n\nYour previous answer used items that are not in EVIDENCE: "
                    f"numbers {issues['unverified_numbers']}, tickers {issues['unexpected_tickers']}. Rewrite using only values from EVIDENCE.")
    return narr, used, issues


def advise(res: dict, results: dict | None = None, cfg=None, *, api_key: str | None = None, model: str | None = None, generate=None,
           today: date | None = None, use_llm: bool = True, esg=None) -> dict:
    """One call: evidence -> (Gemini | template) narrative.  `generate(system, user) -> (text, model)` can be injected (tests)."""
    ev = build_evidence(res, results, cfg, today, esg)
    ev["universe_tickers"] = list(res["tickers"])
    key = resolve_api_key(api_key)
    model = model or os.environ.get("GEMINI_MODEL") or DEFAULT_MODEL
    source, used, narr, err, issues = "template", None, None, None, {"unverified_numbers": [], "unexpected_tickers": []}
    if use_llm and (generate is not None or key):
        try:
            narr, used, issues = _llm_narrative(ev, key, model, generate)
            source = "gemini"
        except Exception as e:
            err = _scrub(str(e), key)
    if narr is None:
        narr = template_narrative(ev)
    evidence = {k: v for k, v in ev.items() if k != "universe_tickers"}
    return {"source": source, "model": used, "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "pick": ev["pick"],
            "runner_ups": ev["runner_ups"], "narrative": narr, "fixed_notes": ev["fixed_notes"], "warnings": issues, "llm_error": err,
            "disclaimer": DISCLAIMER, "evidence": evidence}


# ----------------------------------------------------------------------------- output
def to_markdown(adv: dict) -> str:
    p, n, e = adv["pick"], adv["narrative"], adv["evidence"]["run"]
    L = [f"# Portfolio advisor summary", "", f"> {adv['disclaimer']}", ""]
    if e["stale"]:
        L += [f"> **STALE DATA** - decision date {e['decision_date']} ({e['days_since_decision_date']} days before {e['today']}). Demonstration only.", ""]
    L += [f"## {n['headline']}", "", "| ticker | consensus weight | methods holding | quantum mean | classical mean | exp. return 21d | vol 21d | ESG pct |", "|---|---|---|---|---|---|---|---|"]
    nm = len(e["methods_in_consensus"])
    pc = lambda x: "n/a" if x is None else _pct(x, 2)
    nn = lambda x: "n/a" if x is None else x
    for c in [p] + adv["runner_ups"]:
        L.append(f"| {c['ticker']} | {_pct(c['consensus_weight'])} | {c['methods_holding']}/{nm} | {nn(c['quantum_mean_weight'])} | {nn(c['classical_mean_weight'])} | "
                 f"{pc(c['exp_return_21d'])} | {pc(c['vol_21d'])} | {nn(c['esg_percentile'])} |")
    L += ["", f"Pick confidence: **{p['confidence']}**" + (" (capped at low because the data is stale)" if e["stale"] else "")
          + f"; quantum and classical optimisers agree on the top stock: **{p['quantum_classical_agree']}**.", ""]
    titles = {"what_we_did": "What we did", "how_it_works": "How it works", "fault_tolerance_findings": "Fault-tolerance and mitigation findings",
              "honest_caveats": "Caveats", "what_happens_next": "What happens next"}
    for k in SECTIONS:
        L += [f"## {titles[k]}", ""] + [f"- {s}" for s in n[k]] + [""]
    L += ["## Fixed notes (added by code)", ""] + [f"- {s}" for s in adv["fixed_notes"]] + [""]
    w = adv["warnings"]
    if w["unverified_numbers"] or w["unexpected_tickers"]:
        L += ["## Unverified items in the narrative", "", f"Numbers not found in the evidence: {w['unverified_numbers']}; tickers not in the evidence: {w['unexpected_tickers']}.", ""]
    src = f"Narrative: Gemini ({adv['model']})." if adv["source"] == "gemini" else "Narrative: offline template" + (f" (Gemini unavailable: {adv['llm_error']})" if adv["llm_error"] else " (no API key).")
    L += [f"*{src} Table values and all numbers come from the optimiser code. Generated {adv['generated_at']}.*", ""]
    return "\n".join(L)
