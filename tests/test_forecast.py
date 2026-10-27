"""H1: forecast_returns interface (no look-ahead), time-ordered split, walk-forward bank."""
import numpy as np, pandas as pd, pytest

pytest.importorskip("tensorflow")
from stock_prediction.forecast import forecast_returns, LookAheadError, ModelBank, LSTMForecaster
from stock_prediction.model import HParams, train_pooled, _samples, prepare_panel


def _prices(n=900, k=4, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2015-01-01", periods=n)
    return pd.DataFrame(100 * np.exp(np.cumsum(rng.normal(3e-4, 0.01, (n, k)), 0)), index=idx, columns=list("ABCD"))


@pytest.fixture(scope="module")
def model_and_prices():
    px = _prices()
    hp = HParams(seq_len=20, units=8, dense=8, epochs=1, stride=6, batch=64)
    return train_pooled(px, px.index[599], hp, frac_val=0.2, seed=1), px


def test_forecast_uses_only_data_before_as_of(model_and_prices):
    m, px = model_and_prices
    as_of = px.index[700]
    base = forecast_returns(list("ABCD"), as_of, px, m)
    px2 = px.copy()
    px2.loc[px2.index >= as_of] *= 5.0                         # wreck every price on/after as_of
    assert np.allclose(base, forecast_returns(list("ABCD"), as_of, px2, m))
    px3 = px.copy()
    px3.loc[px3.index == px.index[699]] *= 1.3                  # change the last bar BEFORE as_of -> forecast must react
    assert not np.allclose(base, forecast_returns(list("ABCD"), as_of, px3, m))


def test_model_trained_on_or_after_as_of_is_refused(model_and_prices):
    m, px = model_and_prices
    with pytest.raises(LookAheadError):
        forecast_returns(["A"], m.train_end, px, m)
    with pytest.raises(LookAheadError):
        forecast_returns(["A"], px.index[300], px, m)


def test_unknown_ticker_is_nan(model_and_prices):
    m, px = model_and_prices
    out = forecast_returns(["A", "ZZZ"], px.index[700], px, m)
    assert np.isfinite(out[0]) and np.isnan(out[1])


def test_split_is_time_ordered_and_purged(model_and_prices):
    _, px = model_and_prices
    prep = prepare_panel(px)
    t_val = px.index[480]
    _, _, d_tr, _ = _samples(prep, 20, px.index[0], t_val, t_val - pd.Timedelta(days=1), 3)
    _, _, d_va, _ = _samples(prep, 20, t_val, None, px.index[599], 3)
    assert max(d_tr) < t_val <= min(d_va)
    assert pd.Timestamp(max(d_tr)) + pd.offsets.BDay(21) < t_val + pd.Timedelta(days=1)     # train labels end before the val period starts


def test_model_bank_picks_latest_strictly_earlier(model_and_prices, tmp_path):
    m, px = model_and_prices
    m.save(str(tmp_path), "wf_2016-12-30"); m.save(str(tmp_path), "wf_2017-06-30")
    bank = ModelBank(str(tmp_path), "wf_")
    # saved metadata keeps the REAL train_end, but selection is by file cut-off date; check strictness on the file dates
    assert bank.select("2017-07-01").name == "wf_2017-06-30"
    assert bank.select("2017-06-30").name == "wf_2016-12-30"
    with pytest.raises(LookAheadError):
        bank.select("2016-12-30")


def test_lstm_forecaster_memoises(model_and_prices):
    m, px = model_and_prices
    f = LSTMForecaster(m, px)
    a = f(px.index[700], list("AB")); b = f(px.index[700], list("AB"))
    assert a is b
