"""Background workers so a UI never blocks on a QAOA solve (G9).  Pure threading, no Streamlit imports (unit-testable)."""
from __future__ import annotations
import threading, time, traceback
import pandas as pd

from .live import LiveOptimizer


class Job:
    """Run fn(*a, **k) in a daemon thread; poll .state / .result / .error."""
    def __init__(self, fn, *a, **k):
        self.state, self.result, self.error, self.t0, self.t1 = "running", None, None, time.time(), None
        def run():
            try:
                self.result = fn(*a, **k)
                self.state = "done"
            except Exception:
                self.error = traceback.format_exc(limit=3)
                self.state = "error"
            self.t1 = time.time()
        self.thread = threading.Thread(target=run, daemon=True)
        self.thread.start()

    @property
    def seconds(self) -> float:
        return (self.t1 or time.time()) - self.t0


class StreamJob:
    """Steps a LiveOptimizer (replay or feed) in a thread; `snapshot()` returns a consistent copy for plotting."""
    def __init__(self, lo: LiveOptimizer, n_ticks: int, delay: float = 0.0, poll_wait: float = 1.0):
        self.lo, self.n, self.delay, self.poll_wait = lo, n_ticks, delay, poll_wait
        self.lock = threading.Lock()
        self.t, self.state, self.error, self.stop_flag, self.last = 0, "running", None, False, {}
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        try:
            while self.t < self.n and not self.stop_flag:
                with self.lock:
                    out = self.lo.step()
                if out.get("done"):
                    break
                if out.get("waiting"):
                    time.sleep(self.poll_wait); continue
                self.t += 1; self.last = out
                time.sleep(self.delay)
            self.state = "stopped" if self.stop_flag else "done"
        except Exception:
            self.error = traceback.format_exc(limit=3); self.state = "error"

    def stop(self):
        self.stop_flag = True

    def snapshot(self) -> dict:
        with self.lock:
            lo = self.lo
            return {"pnl": pd.DataFrame(lo.pnl, index=lo.dates), "target": {k: v.copy() for k, v in lo.target.items()},
                    "info": dict(lo.last_info), "tickers": list(lo.m.returns.columns), "t": self.t, "n": self.n, "state": self.state,
                    "last": dict(self.last), "error": self.error}
