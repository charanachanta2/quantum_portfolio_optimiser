"""The combined Streamlit app renders every tab offline, and 'Solve now' produces weights from the background worker."""
import time
import pytest
pytest.importorskip("streamlit")
pytest.importorskip("tensorflow")
from streamlit.testing.v1 import AppTest


def test_combined_app_renders_and_solves():
    at = AppTest.from_file("../app.py", default_timeout=240)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    assert len(at.tabs) == 5
    assert any("not financial advice" in w.value for w in at.warning)          # G10 safety label
    btn = next(b for b in at.button if b.label == "Solve now")
    btn.click().run()
    job = at.session_state["job"]
    t0 = time.time()
    while job.state == "running" and time.time() - t0 < 120:
        time.sleep(0.2)
    assert job.state == "done", job.error
    assert set(job.result["weights"]) >= {"qaoa_noisy_mitigated", "exact_qubo", "equal_weight_full"}
    at.run()
    assert not at.exception


def test_advisor_panel_in_combined_app(monkeypatch):
    """Solve now -> 'Generate advice' in the AI advisor panel gives a pick (offline template: no key set, so no network call)."""
    for k in ("GEMINI_API_KEY", "GOOGLE_API_KEY"):
        monkeypatch.delenv(k, raising=False)
    at = AppTest.from_file("../app.py", default_timeout=240)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    next(b for b in at.button if b.label == "Solve now").click().run()
    job = at.session_state["job"]
    t0 = time.time()
    while job.state == "running" and time.time() - t0 < 120:
        time.sleep(0.2)
    assert job.state == "done", job.error
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    btn = next(b for b in at.button if b.label.startswith("Generate advice"))
    assert "offline template" in btn.label
    btn.click().run()
    assert not at.exception, [e.value for e in at.exception]
    pick = next(m for m in at.metric if m.label == "Top consensus stock")
    assert pick.value in job.result["tickers"]
    assert any("Narrative from the offline template" in c.value for c in at.caption)
