"""Streamlit panel for the Gemini advisor (kept out of app.py so it can be tested without TensorFlow).
`render_panel(st, ...)` is called from the Live tab after 'Solve now' has finished."""
from __future__ import annotations
import pandas as pd

from . import advisor


def show_advice(st, adv: dict) -> None:
    p, n, e = adv["pick"], adv["narrative"], adv["evidence"]["run"]
    nm = len(e["methods_in_consensus"])
    if e["stale"]:
        st.error(f"STALE DATA: decision date {e['decision_date']} is {e['days_since_decision_date']} days old. "
                 "Switch the sidebar to 'Live (Yahoo Finance)' for a current pick; this one is a demonstration.")
    st.markdown(f"#### {n['headline']}")
    c = st.columns(4)
    c[0].metric("Top consensus stock", p["ticker"], f"{p['consensus_weight']:.1%} avg weight")
    c[1].metric("Held by methods", f"{p['methods_holding']}/{nm}")
    c[2].metric("Quantum = classical top?", "yes" if p["quantum_classical_agree"] else "no")
    c[3].metric("Pick confidence", p["confidence"])
    tab = pd.DataFrame([{k: x[k] for k in ("ticker", "consensus_weight", "methods_holding", "quantum_mean_weight", "classical_mean_weight",
                                           "exp_return_21d", "vol_21d", "esg_percentile")} for x in [p] + adv["runner_ups"]]).set_index("ticker")
    st.dataframe(tab, width="stretch")
    titles = {"what_we_did": "What we did", "how_it_works": "How it works", "fault_tolerance_findings": "Fault-tolerance and mitigation findings",
              "honest_caveats": "Caveats", "what_happens_next": "What happens next"}
    for k in advisor.SECTIONS:
        st.markdown(f"**{titles[k]}**\n\n" + "\n".join(f"- {s}" for s in n[k]))
    for s in adv["fixed_notes"]:
        st.caption(s)
    w = adv["warnings"]
    if w["unverified_numbers"] or w["unexpected_tickers"]:
        st.warning(f"The narrative contains items not found in the evidence - treat with care. Numbers: {w['unverified_numbers']}; tickers: {w['unexpected_tickers']}.")
    if adv["source"] == "gemini":
        st.caption(f"Narrative written by Gemini ({adv['model']}). The ranking table and every number come from the optimiser code.")
    else:
        why = f"Gemini call failed: {adv['llm_error']}" if adv["llm_error"] else "no Gemini API key set"
        st.caption(f"Narrative from the offline template ({why}).")
    st.download_button("Download summary (Markdown)", advisor.to_markdown(adv), file_name="advice.md", mime="text/markdown")
    with st.expander("Evidence sent to Gemini (JSON)"):
        st.json(adv["evidence"])


def render_panel(st, job, results: dict, cfg, api_key: str | None, model: str | None, esg=None) -> None:
    if job is None or job.state != "done":
        st.info("Run **Solve now** first. The advisor ranks the stocks from that solve and writes the summary.")
        return
    key = advisor.resolve_api_key(api_key)
    label = f"Generate advice with Gemini ({model or advisor.DEFAULT_MODEL})" if key else "Generate advice (offline template - no Gemini key set)"
    if st.button(label):
        with st.spinner("Ranking stocks and writing the summary ..."):
            adv = advisor.advise(job.result, results, cfg, api_key=key, model=model or None, esg=esg)
        st.session_state["advice"] = (id(job.result), adv)
    cur = st.session_state.get("advice")
    if cur and cur[0] == id(job.result):
        show_advice(st, cur[1])
