"""Item 8: capture dashboard screenshots with Playwright.
    pip install playwright streamlit && python -m playwright install chromium
    python scripts/screenshots.py --real --out results_real          # (omit --real for synthetic)
Starts `streamlit run dashboard/app.py` headless, clicks through the sidebar, saves PNGs (idle, mid-stream, final).
STATUS: executed on the author's machine up to the data-loading step; --real requires data/esg.csv. If a selector fails, open the page
with `--headed`, and adjust the label strings below (they must match dashboard/app.py)."""
import argparse, os, subprocess, sys, time, urllib.request


def wait_http(url, timeout=90):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            urllib.request.urlopen(url, timeout=2); return True
        except Exception:
            time.sleep(1)
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", action="store_true"); ap.add_argument("--out", default="results"); ap.add_argument("--port", type=int, default=8599)
    ap.add_argument("--headed", action="store_true"); ap.add_argument("--ticks", type=int, default=80)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if a.real:                                              # fail fast instead of waiting 15 min on an error page
        miss = [f for f in ("data/esg.csv", "data/raw") if not os.path.exists(os.path.join(root, f))]
        if miss:
            sys.exit(f"--real needs {miss} (run scripts/fetch_data.py and provide/build data/esg.csv first), "
                     "or run without --real for the synthetic-data screenshots.")
    proc = subprocess.Popen([sys.executable, "-m", "streamlit", "run", "dashboard/app.py", "--server.headless", "true",
                             "--server.port", str(a.port), "--browser.gatherUsageStats", "false"], cwd=root,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        if not wait_http(f"http://localhost:{a.port}/_stcore/health"):
            sys.exit("streamlit did not start")
        from playwright.sync_api import sync_playwright
        with sync_playwright() as pw:
            br = pw.chromium.launch(headless=not a.headed)
            page = br.new_page(viewport={"width": 1500, "height": 950})
            page.goto(f"http://localhost:{a.port}"); page.wait_for_selector("text=Fault-tolerant quantum portfolio optimizer", timeout=60000)
            tag = "real" if a.real else "synthetic"
            if a.real:
                page.get_by_text("Real (data/raw + data/esg.csv)").click(); page.wait_for_timeout(4000)
                if page.get_by_text("Could not load the selected data").count():
                    msg = page.get_by_text("Could not load the selected data").first.inner_text()
                    br.close(); sys.exit("Dashboard could not load the real data:\n" + msg)
            page.screenshot(path=f"{a.out}/dashboard_{tag}_idle.png")
            page.get_by_role("button", name="Start stream").click()
            page.wait_for_selector("text=tick", timeout=120000)   # raises a clear Playwright timeout if the stream never starts; page.wait_for_timeout(15000)
            page.screenshot(path=f"{a.out}/dashboard_{tag}_midstream.png")
            print("waiting for the replay to finish (up to 15 min; Ctrl+C to abort)..."); page.wait_for_selector("text=Replay finished", timeout=900000); page.wait_for_timeout(1500)
            page.screenshot(path=f"{a.out}/dashboard_{tag}_final.png", full_page=True)
            br.close()
        print("saved", [f for f in os.listdir(a.out) if f.startswith("dashboard_")])
    finally:
        proc.terminate()


if __name__ == "__main__":
    main()
