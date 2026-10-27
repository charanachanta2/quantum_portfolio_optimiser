import os, pytest
def test_dashboard_runs():
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(os.path.join(os.path.dirname(__file__), "..", "dashboard", "app.py"), default_timeout=120)
    at.run(); assert not at.exception
    at.slider(key=None) if False else None
    for b in at.button:
        if "Start" in b.label:
            b.click()
    at.run(); assert not at.exception
