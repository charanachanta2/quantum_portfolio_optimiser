"""F2/F3/F5: mu_source and risk_source switches in make_instance."""
import numpy as np, pytest
from dataclasses import replace
from qpo.config import Config
from qpo.data import synthetic_market
from qpo.backtest import make_instance


@pytest.fixture(scope="module")
def mkt():
    return synthetic_market(n_assets=12)


class Fake:
    """forecaster(date, tickers) -> 21d returns; records the dates it was asked about."""
    def __init__(self, val, conf=0.25): self.val, self.conf, self.dates = val, conf, []
    def __call__(self, d, tickers): self.dates.append(d); return np.full(len(tickers), self.val) + np.arange(len(tickers)) * 1e-4
    def confidence(self, d): return self.conf


def test_default_is_historical_and_needs_no_forecaster(mkt):
    cfg = Config(); w = np.full(12, 1 / 12)
    a = make_instance(mkt, 600, w, cfg); b = make_instance(mkt, 600, w, cfg, Fake(0.5))
    assert np.allclose(a.mu, b.mu) and a.lam == b.lam and np.allclose(a.qubo.Q, b.qubo.Q)


def test_lstm_and_blend_replace_mu_only(mkt):
    w = np.full(12, 1 / 12); base = Config(rebalance_days=21)
    h = make_instance(mkt, 600, w, base)
    f = Fake(0.02)
    l = make_instance(mkt, 600, w, replace(base, mu_source="lstm"), f)
    bl = make_instance(mkt, 600, w, replace(base, mu_source="blend", lstm_blend=0.5), f)
    assert np.array_equal(h.sub, l.sub)                         # asset selection (and caps, costs, ESG filter) untouched
    assert np.allclose(l.mu, 0.02 + (np.searchsorted(h.elig, l.sub)) * 1e-4, atol=1e-12)
    assert np.allclose(bl.mu, 0.5 * l.mu + 0.5 * h.mu)
    assert np.allclose(l.Sigma, h.Sigma) and l.caps == h.caps and l.lam == h.lam
    assert f.dates and all(d == mkt.returns.index[600] for d in f.dates)    # asked at the decision date, nothing later


def test_lstm_without_forecaster_raises(mkt):
    with pytest.raises(ValueError):
        make_instance(mkt, 600, np.full(12, 1 / 12), Config(mu_source="lstm"))


def test_nan_forecast_falls_back_to_historical(mkt):
    w = np.full(12, 1 / 12)
    h = make_instance(mkt, 600, w, Config())
    l = make_instance(mkt, 600, w, Config(mu_source="lstm"), lambda d, t: np.full(len(t), np.nan))
    assert np.allclose(h.mu, l.mu)


def test_fuzzy_risk_changes_lambda_with_confidence(mkt):
    w = np.full(12, 1 / 12)
    base = make_instance(mkt, 600, w, Config())
    hi = make_instance(mkt, 600, w, Config(risk_source="vix+fuzzy"), Fake(0.0, conf=0.05))
    lo = make_instance(mkt, 600, w, Config(risk_source="vix+fuzzy", kappa_risk=0.5), Fake(0.0, conf=0.95))
    assert hi.lam[0] >= lo.lam[0] and hi.lam[0] > 0 and base.lam[0] > 0
