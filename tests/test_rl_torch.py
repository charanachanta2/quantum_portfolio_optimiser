"""Skipped (not failed) when torch is missing.  These tests are the acceptance gate for qpo/rl_torch.py."""
import numpy as np, pytest


def _train_rets():
    from qpo.data import synthetic_market
    m = synthetic_market(n_assets=8, start="2005-01-01", end="2008-12-31", seed=3)
    return m.returns.values


def test_act_is_on_simplex_and_matches_es_interface():
    pytest.importorskip("torch")
    from qpo.rl_torch import TorchAgent
    R = _train_rets()
    ag = TorchAgent(seed=0)
    w = ag.act(R[-126:], np.full(8, 1 / 8))
    assert w.shape == (8,) and abs(w.sum() - 1) < 1e-9 and (w >= 0).all()


def test_training_improves_in_sample_reward_and_is_reproducible():
    pytest.importorskip("torch")
    from qpo.rl_torch import TorchAgent
    R = _train_rets()
    a = TorchAgent(seed=1)
    before = a.evaluate(R)
    a.fit(R, epochs=80)
    after = a.evaluate(R)
    assert np.isfinite(after) and after > before
    b = TorchAgent(seed=1).fit(R, epochs=80)                 # same seed -> same policy
    assert np.allclose(a.act(R[-126:], np.full(8, 1 / 8)), b.act(R[-126:], np.full(8, 1 / 8)), atol=1e-6)
