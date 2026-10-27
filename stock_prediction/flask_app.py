"""Flask UI (routes only).  Run from the project root:  python -m stock_prediction.flask_app
The model is trained OFFLINE (scripts/train_predictor.py) and loaded once; requests never train anything (E2)."""
from __future__ import annotations
import functools, os
import numpy as np
import pandas as pd
import plotly.graph_objs as go
from flask import Flask, render_template, request

from . import data as D
from .model import Predictor
from .service import analyse_symbol, prediction_vs_actual

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.environ.get("SPS_MODEL_DIR", os.path.join(ROOT, "models"))
MODEL_NAME = os.environ.get("SPS_MODEL_NAME", "deployed")
app = Flask(__name__)


@functools.lru_cache(maxsize=1)
def get_predictor() -> Predictor:
    return Predictor.load(MODEL_DIR, MODEL_NAME)


def _charts(symbol, close, res, pva):
    f1 = go.Figure(go.Scatter(x=close.index[-250:], y=close.iloc[-250:], name="Close"))
    f1.add_trace(go.Scatter(x=res["dates"], y=res["path"], name="7-day forecast", line=dict(dash="dot")))
    f1.add_trace(go.Scatter(x=list(res["dates"]) + list(res["dates"])[::-1], y=list(res["hi"]) + list(res["lo"])[::-1], fill="toself",
                            name="90% band (validation residuals)", mode="lines", line=dict(width=0), opacity=0.25))
    f1.update_layout(title=f"{symbol}: price history and 7-day forecast", xaxis_title="Date", yaxis_title="Price")
    f2 = go.Figure()
    f2.add_trace(go.Scatter(x=pva.index, y=pva["actual_next"], name="Actual"))
    f2.add_trace(go.Scatter(x=pva.index, y=pva["pred_next"], name="Predicted (next day)"))
    f2.add_trace(go.Scatter(x=pva.index, y=pva["naive_next"], name="Naive (tomorrow = today)", line=dict(dash="dot")))
    f2.update_layout(title="Next-day prediction vs actual (points after the training cut-off are out-of-sample)", xaxis_title="Date", yaxis_title="Price")
    return f1.to_html(full_html=False), f2.to_html(full_html=False)


@app.route("/", methods=["GET", "POST"])
def index():
    ctx = dict(companies=D.symbols_table().to_dict("records"), stock_chart=None, pred_chart=None, rmse=None, naive_rmse=None, skill=None,
               split=None, train_end=None, insights=None, forecast=None, confidence=None, risk=None, error=None, accuracy=None)
    if request.method == "POST":
        symbol = request.form["symbol"].strip().upper()
        try:
            ohlcv = D.fetch_ohlcv(symbol)
            if len(ohlcv) < 100:
                raise ValueError("Not enough data.")
            close = ohlcv["Close"]
            pr = get_predictor()
            res = analyse_symbol(symbol, close, pr, volume=ohlcv.get("Volume"))
            pva = prediction_vs_actual(close, pr, 250)
            ctx["stock_chart"], ctx["pred_chart"] = _charts(symbol, close, res, pva)
            last = res["last"]
            ctx.update(rmse=round(res["rmse_h1_model"] * last, 3) if res["rmse_h1_model"] is not None else None,
                       naive_rmse=round(res["rmse_h1_naive"] * last, 3) if res["rmse_h1_naive"] is not None else None,
                       skill=res["skill_h1"], split=res["metrics_split"], train_end=res["model_train_end"],
                       accuracy=round(100 * res["dir_acc_h1"], 1) if res["dir_acc_h1"] is not None else None,
                       insights={"current_price": round(last, 2), "predicted_price": round(float(res["path"][-1]), 2),
                                 "profit": round(float(res["path"][-1]) - last, 2), "profit_percent": round(res["exp_ret_7d_pct"], 2),
                                 "signal": res["signal"]},
                       forecast=[round(float(x), 2) for x in res["path"]], confidence=res["confidence_label"],
                       risk=f"{res['risk']['label']} (fuzzy score {res['risk']['score']:.2f}, ann. vol {res['risk']['ann_vol']:.0%})")
        except FileNotFoundError:
            ctx["error"] = "No trained model found. Train it offline first:  python scripts/train_predictor.py"
        except Exception as e:                                           # network, unknown symbol, short history
            ctx["error"] = str(e)
    return render_template("index.html", **ctx)


if __name__ == "__main__":
    app.run(debug=False)
