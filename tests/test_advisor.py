"""Gemini advisor: deterministic pick, Gemini REST client (mocked - no network), validation guardrails, offline fallback, Streamlit panel."""
import io
import json
import os
import types
import urllib.error
from datetime import date

import numpy as np
import pandas as pd
import pytest

from qpo import advisor
from qpo.config import Config
from qpo.data import synthetic_market
from qpo.live import solve_now

METHODS = ["exact_qubo", "sa_qubo", "miqp_sub", "cvar_sub", "equal_weight_full"]
TODAY = date(2026, 10, 27)


@pytest.fixture(autouse=True)
def _no_ambient_key(monkeypatch):
    for k in ("GEMINI_API_KEY", "GOOGLE_API_KEY", "GEMINI_MODEL"):
        monkeypatch.delenv(k, raising=False)


@pytest.fixture(scope="module")
def solved():
    m = synthetic_market("2012-01-01", "2018-12-31", n_assets=8, seed=1)
    cfg = Config(n_assets=3, shots=64, p_max=1, opt_maxiter=5)
    return solve_now(m, cfg, METHODS), cfg, m


def fake_results(real=True, n_sig=0):
    return {"meta": {"data_source": "yahoo+fred+esg" if real else "synthetic", "data_meta": {"esg_provenance": ["test"], "esg_missing": [], "dropped_macro": []}},
            "is_real": real,
            "summary_GFC_2008": pd.DataFrame({"sharpe": {"qaoa_ideal": 0.1, "qaoa_logical": 0.2, "exact_qubo": 0.3, "equal_weight_full": 0.05}}),
            "signif_GFC_2008": pd.DataFrame({"p_boot_holm": [1.0] * (40 - n_sig) + [0.01] * n_sig}),
            "ablation": pd.DataFrame({"est_err": {"raw": 0.0924, "ZNE": 0.0326, "PEC": 0.0381}, "overhead": {"raw": 1.0, "ZNE": 57.0, "PEC": 24.32},
                                      "q_eff": {"raw": 0.0464, "ZNE": 0.0464, "PEC": 0.0464}}),
            "scaling_fits": pd.DataFrame({"log2_slope_per_qubit": {"N99_qaoa": 0.846, "SA_evals_median": 0.42}}),
            "resource_corners": pd.DataFrame([{"circuit": "small n=16", "p_phys": 0.001, "scenario": "baseline", "distance": 15, "physical_qubits": 75576,
                                               "time_per_shot_ms": 326.6, "distill_levels": 1, "note": np.nan}]),
            "qec_fit": {"p_th": 0.0098, "n_points": 9}}


def hand_res(weights, tickers=("AAA", "BBB", "CCC"), mu=(0.01, 0.02, 0.03), sd=(0.1, 0.1, 0.1)):
    return {"date": pd.Timestamp("2026-10-26"), "tickers": list(tickers), "elig": list(tickers), "mu_elig": list(mu),
            "Sigma_elig": np.diag(np.square(sd)).tolist(), "weights": {k: np.array(v, float) for k, v in weights.items()},
            "info": {k: {} for k in weights}, "sub": list(tickers), "n_qubits": 12, "lam": [3.0, 4.5], "caps": [3, 3], "mu_source": "historical", "risk_source": "vix"}


# ----------------------------------------------------------------------------- deterministic pick
def test_pick_is_consensus_and_excludes_equal_weight():
    res = hand_res({"exact_qubo": [0.5, 0.5, 0.0], "qaoa_ideal": [0.5, 0.0, 0.5], "equal_weight_full": [1 / 3] * 3})
    ev = advisor.build_evidence(res, None, None, TODAY)
    assert "equal_weight_full" not in ev["run"]["methods_in_consensus"]
    assert ev["pick"]["ticker"] == "AAA" and ev["pick"]["consensus_weight"] == 0.5 and ev["pick"]["methods_holding"] == 2
    assert ev["pick"]["quantum_mean_weight"] == 0.5 and ev["run"]["quantum_methods"] == ["qaoa_ideal"]


def test_tie_broken_by_return_per_vol_then_ticker():
    res = hand_res({"exact_qubo": [0.5, 0.5, 0.0], "sa_qubo": [0.5, 0.5, 0.0]})          # AAA and BBB tie; BBB has higher mu/sigma
    assert advisor.build_evidence(res, None, None, TODAY)["pick"]["ticker"] == "BBB"
    res = hand_res({"exact_qubo": [0.5, 0.5, 0.0]}, mu=(0.02, 0.02, 0.02))                  # full tie -> alphabetical
    assert advisor.build_evidence(res, None, None, TODAY)["pick"]["ticker"] == "AAA"


def test_stale_data_caps_confidence_and_adds_note():
    res = hand_res({"exact_qubo": [1, 0, 0], "sa_qubo": [1, 0, 0], "miqp_sub": [1, 0, 0]})
    fresh = advisor.build_evidence(res, None, None, date(2026, 10, 28))
    assert fresh["pick"]["confidence"] == "high" and not fresh["run"]["stale"]
    old = advisor.build_evidence(res, None, None, date(2027, 3, 1))
    assert old["pick"]["confidence"] == "low" and old["run"]["stale"] and any("STALE" in n for n in old["fixed_notes"])


def test_evidence_includes_fault_tolerance_backtests_and_significance_note():
    res = hand_res({"exact_qubo": [1, 0, 0], "qaoa_logical": [1, 0, 0]})
    ev = advisor.build_evidence(res, fake_results(), None, TODAY)
    assert ev["fault_tolerance"]["mitigation_ablation"]["ZNE"]["overhead"] == 57.0
    assert ev["fault_tolerance"]["resource_estimates_baseline"][0]["physical_qubits"] == 75576
    assert ev["backtests"]["GFC_2008"]["significant_diffs"] == 0 and ev["backtests"]["GFC_2008"]["total_comparisons"] == 40
    assert any("0 of 40" in n and "no quantum advantage" in n for n in ev["fixed_notes"])
    assert not any("SYNTHETIC" in n for n in ev["fixed_notes"])
    assert any("not marked as real" in n for n in advisor.build_evidence(res, fake_results(real=False), None, TODAY)["fixed_notes"])


def test_real_solve_end_to_end_offline(solved):
    res, cfg, m = solved
    adv = advisor.advise(res, fake_results(), cfg, today=TODAY, esg=m.esg)
    assert adv["source"] == "template" and adv["llm_error"] is None
    assert adv["pick"]["ticker"] in res["tickers"] and adv["pick"]["consensus_weight"] > 0
    assert adv["warnings"] == {"unverified_numbers": [], "unexpected_tickers": []}
    md = advisor.to_markdown(adv)
    assert adv["pick"]["ticker"] in md and "not financial advice" in md.lower() and "What happens next" in md


@pytest.mark.skipif(not os.path.exists("results_real/meta.json"), reason="results_real/ not present")
def test_template_numbers_pass_own_checker_on_real_results(solved):
    from qpo import dash_data as DD
    res, cfg, _ = solved
    results = DD.load_results("results_real")
    ev = advisor.build_evidence(res, results, cfg, TODAY)
    ev["universe_tickers"] = list(res["tickers"])
    assert ev["fault_tolerance"]["mitigation_ablation"] and ev["backtests"]
    assert advisor.check_narrative(advisor.template_narrative(ev), ev) == {"unverified_numbers": [], "unexpected_tickers": []}


# ----------------------------------------------------------------------------- Gemini REST client (mocked)
class FakeResp:
    def __init__(self, payload):
        self._b = json.dumps(payload).encode()

    def read(self):
        return self._b

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def gem_payload(obj):
    return {"candidates": [{"content": {"parts": [{"text": obj if isinstance(obj, str) else json.dumps(obj)}]}}]}


def http_error(code, msg):
    return urllib.error.HTTPError("https://x", code, "err", {}, io.BytesIO(json.dumps({"error": {"message": msg}}).encode()))


def test_request_shape_and_key_handling(monkeypatch):
    seen = {}

    def fake(req, timeout=None):
        seen.update(url=req.full_url, key=req.get_header("X-goog-api-key"), body=json.loads(req.data), method=req.get_method())
        return FakeResp(gem_payload({"ok": True}))
    monkeypatch.setattr(advisor, "_urlopen", fake)
    text, used = advisor.gemini_generate("sys", "user", api_key="SECRET123", model="gemini-3.8-flash")
    assert used == "gemini-3.8-flash" and json.loads(text) == {"ok": True}
    assert seen["url"] == "https://generativelanguage.googleapis.com/v1beta/models/gemini-3.8-flash:generateContent" and seen["method"] == "POST"
    assert seen["key"] == "SECRET123" and "SECRET123" not in seen["url"]                      # header, never in the URL
    b = seen["body"]
    assert b["systemInstruction"]["parts"][0]["text"] == "sys" and b["contents"][0]["parts"][0]["text"] == "user"
    assert b["generationConfig"]["responseMimeType"] == "application/json"


def test_404_falls_back_to_next_model(monkeypatch):
    urls = []

    def fake(req, timeout=None):
        urls.append(req.full_url)
        if "gemini-9-nope" in req.full_url:
            raise http_error(404, "model not found")
        return FakeResp(gem_payload({"ok": 1}))
    monkeypatch.setattr(advisor, "_urlopen", fake)
    _, used = advisor.gemini_generate("s", "u", api_key="k", model="gemini-9-nope")
    assert used == advisor.FALLBACK_MODELS[0] and len(urls) == 2


def test_retries_429_then_succeeds_and_gives_up(monkeypatch):
    monkeypatch.setattr(advisor, "_sleep", lambda s: None)
    calls = {"n": 0}

    def flaky(req, timeout=None):
        calls["n"] += 1
        if calls["n"] < 3:
            raise http_error(429, "quota")
        return FakeResp(gem_payload({"ok": 1}))
    monkeypatch.setattr(advisor, "_urlopen", flaky)
    advisor.gemini_generate("s", "u", api_key="k")
    assert calls["n"] == 3
    monkeypatch.setattr(advisor, "_urlopen", lambda req, timeout=None: (_ for _ in ()).throw(http_error(429, "quota")))
    with pytest.raises(advisor.GeminiError) as e:
        advisor.gemini_generate("s", "u", api_key="k", max_retries=2)
    assert e.value.status == 429


def test_error_message_never_contains_the_key(monkeypatch):
    monkeypatch.setattr(advisor, "_urlopen", lambda req, timeout=None: (_ for _ in ()).throw(http_error(403, "API key SECRET123 not valid")))
    with pytest.raises(advisor.GeminiError) as e:
        advisor.gemini_generate("s", "u", api_key="SECRET123")
    assert "SECRET123" not in str(e.value) and e.value.status == 403


def test_blocked_or_empty_response_raises(monkeypatch):
    monkeypatch.setattr(advisor, "_urlopen", lambda req, timeout=None: FakeResp({"promptFeedback": {"blockReason": "SAFETY"}}))
    with pytest.raises(advisor.GeminiError, match="no candidates"):
        advisor.gemini_generate("s", "u", api_key="k")


# ----------------------------------------------------------------------------- advise(): guardrails and fallback
def _ev_for(res, results=None):
    return advisor.build_evidence(res, results or fake_results(), None, TODAY)


def test_gemini_cannot_change_the_pick_and_foreign_tickers_are_flagged():
    res = hand_res({"exact_qubo": [0.5, 0.5, 0.0], "sa_qubo": [0.5, 0.5, 0.0]}, tickers=("AAPL", "MSFT", "JNJ"))
    res["elig"] = ["AAPL", "MSFT"]; res["mu_elig"] = [0.01, 0.02]; res["Sigma_elig"] = np.diag([0.01, 0.01]).tolist()
    res["sub"] = ["AAPL", "MSFT"]                                                              # JNJ is not on the qubits and nobody holds it
    narr = advisor.template_narrative(_ev_for(res))
    narr["headline"] = "JNJ is the best stock to load up on."          # JNJ is in the universe but not held by any method
    adv = advisor.advise(res, fake_results(), None, generate=lambda s, u: (json.dumps(narr), "gemini-test"), today=TODAY)
    assert adv["source"] == "gemini" and adv["pick"]["ticker"] == "MSFT"                      # code decides, not Gemini
    assert adv["warnings"]["unexpected_tickers"] == ["JNJ"]


def test_fabricated_number_triggers_one_corrective_retry():
    res = hand_res({"exact_qubo": [0.5, 0.5, 0.0], "sa_qubo": [0.5, 0.5, 0.0]})
    good = advisor.template_narrative(_ev_for(res))
    bad = dict(good, headline="Sharpe of 7.7777 and 123456 physical qubits")
    prompts = []

    def gen(s, u):
        prompts.append(u)
        return json.dumps(bad if len(prompts) == 1 else good), "gemini-test"
    adv = advisor.advise(res, fake_results(), None, generate=gen, today=TODAY)
    assert len(prompts) == 2 and "7.7777" in prompts[1] and adv["warnings"]["unverified_numbers"] == []
    assert adv["narrative"]["headline"] == good["headline"]


def test_unfixed_fabrication_is_kept_but_reported():
    res = hand_res({"exact_qubo": [1, 0, 0]})
    bad = dict(advisor.template_narrative(_ev_for(res)), headline="Guaranteed 987654 return")
    adv = advisor.advise(res, fake_results(), None, generate=lambda s, u: (json.dumps(bad), "m"), today=TODAY)
    assert "987654" in adv["warnings"]["unverified_numbers"] and "987654" in advisor.to_markdown(adv)


def test_invalid_json_twice_falls_back_to_template():
    res = hand_res({"exact_qubo": [1, 0, 0]})
    adv = advisor.advise(res, fake_results(), None, generate=lambda s, u: ("not json at all", "m"), today=TODAY)
    assert adv["source"] == "template" and "invalid JSON" in adv["llm_error"]
    assert set(advisor.SECTIONS) <= set(adv["narrative"])


def test_api_failure_falls_back_and_scrubs_key(monkeypatch):
    monkeypatch.setattr(advisor, "_urlopen", lambda req, timeout=None: (_ for _ in ()).throw(http_error(400, "bad key SECRET123")))
    res = hand_res({"exact_qubo": [1, 0, 0]})
    adv = advisor.advise(res, fake_results(), None, api_key="SECRET123", today=TODAY)
    assert adv["source"] == "template" and "SECRET123" not in json.dumps(adv, default=str) and adv["llm_error"]


def test_no_key_means_no_network_call(monkeypatch):
    monkeypatch.setattr(advisor, "_urlopen", lambda *a, **k: (_ for _ in ()).throw(AssertionError("network used without a key")))
    res = hand_res({"exact_qubo": [1, 0, 0]})
    assert advisor.advise(res, fake_results(), None, today=TODAY)["source"] == "template"
    assert advisor.advise(res, fake_results(), None, api_key="k", use_llm=False, today=TODAY)["source"] == "template"


def test_fenced_json_is_accepted():
    res = hand_res({"exact_qubo": [1, 0, 0]})
    narr = advisor.template_narrative(_ev_for(res))
    adv = advisor.advise(res, fake_results(), None, generate=lambda s, u: ("```json\n" + json.dumps(narr) + "\n```", "m"), today=TODAY)
    assert adv["source"] == "gemini"


def test_key_resolution_order(monkeypatch):
    assert advisor.resolve_api_key() is None
    monkeypatch.setenv("GOOGLE_API_KEY", "g"); assert advisor.resolve_api_key() == "g"
    monkeypatch.setenv("GEMINI_API_KEY", "e"); assert advisor.resolve_api_key() == "e"
    assert advisor.resolve_api_key("  x ") == "x"


# ----------------------------------------------------------------------------- Streamlit panel
def test_streamlit_panel_renders_offline(solved):
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    def script():
        import streamlit as st
        from qpo import advisor_ui
        advisor_ui.render_panel(st, st.session_state["job"], st.session_state["results"], st.session_state["cfg"], None, None)
    res, cfg, _ = solved
    at = AppTest.from_function(script, default_timeout=60)
    at.session_state["job"] = types.SimpleNamespace(state="done", result=res)
    at.session_state["results"], at.session_state["cfg"] = fake_results(), cfg
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    btn = next(b for b in at.button if "Generate advice" in b.label)
    assert "offline template" in btn.label                                                     # no key -> says so
    btn.click().run()
    assert not at.exception, [e.value for e in at.exception]
    assert any(m.label == "Top consensus stock" for m in at.metric)
    assert any("not financial advice" in c.value.lower() for c in at.caption)
    idle = AppTest.from_function(script, default_timeout=60)
    idle.session_state["job"] = types.SimpleNamespace(state="running", result=None)
    idle.session_state["results"], idle.session_state["cfg"] = fake_results(), cfg
    idle.run()
    assert any("Solve now" in i.value for i in idle.info)
