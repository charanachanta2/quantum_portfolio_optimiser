"""Stock Prediction System (refactored from the Flask prototype into an importable module).

    features  - scale-free technical features + multi-horizon targets (no look-ahead)
    model     - pooled LSTM, time-ordered train/val/test split, offline training, save/load
    fuzzy     - fuzzy risk and trading-signal inference (E1)
    ga        - real-valued genetic algorithm for LSTM hyper-parameters (E1)
    forecast  - forecast_returns(tickers, as_of_date) -> mu_hat   (the interface used by the optimiser)
    service   - analyse_symbol(): everything the web UIs show for one ticker
    data      - yfinance fetch (TTL cache) + offline snapshot loader
    flask_app - the original Flask UI, now only routes
TensorFlow is imported lazily (only inside model.py) so `import stock_prediction.fuzzy` etc. stay light.
"""
