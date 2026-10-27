"""H3: screenshots of every tab of the combined app (Playwright).
  python -m playwright install chromium
  python scripts/screenshots_combined.py --out results_real/screens_combined
Starts `streamlit run app.py` itself, presses 'Solve now' and 'Start streaming replay', then captures Live / Backtest / Ablation / Resources / About."""
import argparse, os, subprocess, sys, time
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default="results_real/screens_combined"); ap.add_argument("--port", type=int, default=8601)
    ap.add_argument("--wait-solve", type=int, default=90)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    srv = subprocess.Popen([sys.executable, "-m", "streamlit", "run", "app.py", "--server.headless", "true", "--server.port", str(a.port)], cwd=ROOT,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        from playwright.sync_api import sync_playwright
        with sync_playwright() as pw:
            b = pw.chromium.launch(); pg = b.new_page(viewport={"width": 1500, "height": 1100})
            for _ in range(60):
                try:
                    pg.goto(f"http://localhost:{a.port}", timeout=5000); break
                except Exception:
                    time.sleep(1)
            pg.wait_for_selector("text=Prediction panel", timeout=120000); time.sleep(4)
            pg.screenshot(path=f"{a.out}/01_live_idle.png", full_page=True)
            pg.get_by_role("button", name="Solve now").click()
            pg.wait_for_selector("text=Target weights", timeout=a.wait_solve * 1000); time.sleep(3)
            pg.get_by_role("button", name="Start streaming replay").click(); time.sleep(25)
            pg.screenshot(path=f"{a.out}/02_live_solved_streaming.png", full_page=True)
            for i, t in enumerate(["Backtest (2008, 2020)", "Ablation", "Resources", "About"], start=3):
                pg.get_by_role("tab", name=t).click(); time.sleep(3)
                pg.screenshot(path=f"{a.out}/{i:02d}_{t.split(' ')[0].lower()}.png", full_page=True)
            b.close()
    finally:
        srv.terminate()
    print("screenshots ->", a.out)


if __name__ == "__main__":
    main()
