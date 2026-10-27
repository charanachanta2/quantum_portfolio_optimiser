"""PyTorch policy-gradient (REINFORCE) portfolio agent - drop-in sibling of classical.ESAgent.

STATUS: executed on the author's machine (Windows, Python 3.11).  The first run failed with
"RuntimeError: Numpy is not available" because the installed torch wheel was built against NumPy 1.x while
NumPy 2.x was installed.  This module therefore NEVER uses torch's NumPy bridge (Tensor.numpy(),
torch.from_numpy, torch.as_tensor(ndarray)); data crosses the boundary through plain Python lists
(_to_t / _to_np below), so it works with any torch/NumPy combination.  Gate: tests/test_rl_torch.py.

Design (and why it differs from the first draft):
* Policy: per-asset score s_i = MLP(features_i) (weights SHARED across assets -> permutation equivariant, works for
  any number of eligible assets); portfolio = softmax(s).  Exploration: Gaussian noise on the logits,
  a_k = s + sigma*eps_k, w_k = softmax(a_k).  log pi(a_k|s) = sum_i Normal(a_ki; s_i, sigma).  No Dirichlet ->
  no log-prob blow-ups on zero weights.
* Estimator: at every rebalance date (state) draw K actions, reward_k as below, advantage = (r_k - mean_k r)/std_k.
  The baseline is the *same-state* mean (not a mean over different dates, which mostly measures the market
  regime, not action quality).  The state transition (w_prev) follows the deterministic policy mean.
* Reward is IDENTICAL to ESAgent.fit's fitness:  sum(r) - tc*|w - w_prev|_1 - risk*var(r)*rebalance,
  with r the daily portfolio returns over the next `rebalance` days -> the two RL baselines are comparable.
* Interface: .fit(rets, rebalance, ...) -> self ;  .act(window, w_prev) -> np.ndarray on the simplex.
"""
from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn

from .classical import ESAgent


def _to_t(x, device="cpu") -> torch.Tensor:
    """ndarray -> float32 tensor WITHOUT torch's NumPy bridge (works when torch and NumPy ABI-mismatch)."""
    a = np.asarray(x, dtype=np.float64)
    return torch.tensor(a.tolist(), dtype=torch.float32, device=device)


def _to_np(t: torch.Tensor) -> np.ndarray:
    """tensor -> float64 ndarray WITHOUT Tensor.numpy()."""
    return np.asarray(t.detach().double().cpu().tolist(), dtype=np.float64)


class PolicyNet(nn.Module):
    def __init__(self, hidden: int = 16):
        super().__init__()
        self.f = nn.Sequential(nn.Linear(3, hidden), nn.Tanh(), nn.Linear(hidden, 1))

    def forward(self, feats: torch.Tensor) -> torch.Tensor:        # (..., A, 3) -> (..., A)
        return self.f(feats).squeeze(-1)


class TorchAgent:
    def __init__(self, hidden: int = 16, seed: int = 0, sigma: float = 0.5, lr: float = 1e-2, device: str = "cpu"):
        self.seed, self.sigma, self.lr, self.device = seed, sigma, lr, device
        torch.manual_seed(seed)
        self.net = PolicyNet(hidden).to(device)
        self.rng = np.random.default_rng(seed)

    # ---- inference ---------------------------------------------------------
    def _feats(self, window: np.ndarray, w_prev: np.ndarray) -> torch.Tensor:
        return _to_t(ESAgent.features(window, w_prev), self.device)

    @torch.no_grad()
    def act(self, window: np.ndarray, w_prev: np.ndarray) -> np.ndarray:
        s = self.net(self._feats(window, w_prev))
        w = _to_np(torch.softmax(s, -1))
        return w / w.sum()

    # ---- reward (same as ESAgent fitness) ----------------------------------------
    @staticmethod
    def _rewards(R_next: np.ndarray, W: np.ndarray, w_prev: np.ndarray, tc: float, risk: float, rebalance: int):
        """R_next: (reb, A) future daily returns; W: (K, A) candidate weights -> (K,) rewards."""
        r = R_next @ W.T                                            # (reb, K)
        return r.sum(0) - tc * np.abs(W - w_prev[None, :]).sum(1) - risk * r.var(0) * rebalance

    @torch.no_grad()
    def evaluate(self, rets: np.ndarray, rebalance=21, lookback=126, tc=0.001, risk=3.0) -> float:
        """Mean per-rebalance reward of the DETERMINISTIC policy on `rets` (same quantity ESAgent.fit maximises)."""
        A = rets.shape[1]
        w = np.full(A, 1 / A)
        out = []
        for s in range(lookback, len(rets) - rebalance, rebalance):
            wn = self.act(rets[s - lookback:s], w)
            out.append(float(self._rewards(rets[s:s + rebalance], wn[None, :], w, tc, risk, rebalance)[0]))
            w = wn
        return float(np.mean(out))

    # ---- training ---------------------------------------------------------------
    def fit(self, rets: np.ndarray, rebalance: int = 21, lookback: int = 126, tc: float = 0.001, risk: float = 3.0,
            epochs: int = 120, K: int = 64, grad_clip: float = 1.0, verbose: bool = False):
        """REINFORCE with a same-state baseline, antithetic exploration noise and POOLED advantage scaling.

        (The first draft standardised the advantage per state.  That blows up states whose K rewards barely differ,
        so the gradient was dominated by noise and in-sample reward got *worse* with training on every seed we tried.
        Now the advantage is centred per state but scaled by ONE pooled std per epoch, which keeps the true
        relative weight of states, and noise is antithetic (+eps, -eps) with K=64 to cut variance.)"""
        A = rets.shape[1]
        starts = list(range(lookback, len(rets) - rebalance, rebalance))
        if not starts:
            raise ValueError("training window too short for the given lookback/rebalance")
        K = max(2, K + (K % 2))                                                    # antithetic pairs need even K
        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        gen = torch.Generator(device="cpu").manual_seed(self.seed)
        for ep in range(epochs):
            w_prev = np.full(A, 1.0 / A)
            logps, advs = [], []
            for s in starts:
                f = self._feats(rets[s - lookback:s], w_prev)                      # (A,3)
                mean = self.net(f)                                                 # (A,)  carries grad
                e = torch.randn(K // 2, A, generator=gen)
                eps = torch.cat([e, -e]).to(self.device)                           # (K,A) antithetic
                a = mean.detach() + self.sigma * eps                               # (K,A) sampled logits
                logps.append(torch.distributions.Normal(mean, self.sigma).log_prob(a).sum(-1))   # (K,)
                W = _to_np(torch.softmax(a, -1))                                   # (K,A)
                r = self._rewards(rets[s:s + rebalance], W, w_prev, tc, risk, rebalance)
                advs.append(r - r.mean())                                          # same-state baseline
                w_prev = _to_np(torch.softmax(mean.detach(), -1))                  # follow the policy mean
            scale = float(np.concatenate(advs).std()) + 1e-12                      # one pooled scale per epoch
            loss = -sum((lp * _to_t(ad / scale, self.device)).mean() for lp, ad in zip(logps, advs)) / len(starts)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.net.parameters(), grad_clip)
            opt.step()
            if verbose and (ep % 20 == 0 or ep == epochs - 1):
                print(f"  rl_torch epoch {ep + 1}/{epochs}  loss {float(loss):.4f}  train reward {self.evaluate(rets, rebalance, lookback, tc, risk):.5f}")
        return self
