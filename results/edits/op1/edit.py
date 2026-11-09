"""Two-part edit on the noise prediction.

1. time_embedding.linear_1.weight: ridge fit of W' e(t) ~= W e(g(t)), g(t) = t - delta(t),
   delta knots (t: delta): [(0.0, 0.0), (60.0, 13.0), (124.0, 26.0), (166.0, 50.0), (199.0, 70.0), (249.0, 60.0), (332.0, 54.0), (399.0, 54.0), (499.0, 0.0), (999.0, 0.0)]. Zero at t >= 499, so composition steps keep their labels.
2. conv_out.weight *= 0.9885: uniform reduction of predicted noise at every step.
No prompt or step-count dependence.
"""
import math
import torch

KNOTS = [(0.0, 0.0), (60.0, 13.0), (124.0, 26.0), (166.0, 50.0), (199.0, 70.0), (249.0, 60.0), (332.0, 54.0), (399.0, 54.0), (499.0, 0.0), (999.0, 0.0)]
LAM = 10.0
SCALE = 0.9885


def g(t):
    xs = torch.tensor([k[0] for k in KNOTS], dtype=torch.float64)
    ys = torch.tensor([k[1] for k in KNOTS], dtype=torch.float64)
    idx = torch.clamp(torch.searchsorted(xs, t, right=True) - 1, 0, len(xs) - 2)
    frac = (t - xs[idx]) / (xs[idx + 1] - xs[idx])
    return t - (ys[idx] + frac * (ys[idx + 1] - ys[idx]))


def _e(t, dim=320):
    half = dim // 2
    f = torch.exp(-math.log(10000) * torch.arange(half, dtype=torch.float64) / half)
    a = t[:, None] * f[None]
    return torch.cat([torch.cos(a), torch.sin(a)], dim=1)


def apply(w):
    W = w["time_embedding.linear_1.weight"].double()
    ts = torch.arange(1000, dtype=torch.float64)
    E, Eg = _e(ts), _e(g(ts))
    A = E.T @ E + LAM * torch.eye(320, dtype=torch.float64)
    w["time_embedding.linear_1.weight"] = (W + W @ (Eg - E).T @ E @ torch.linalg.inv(A)).float()
    w["conv_out.weight"] *= SCALE
