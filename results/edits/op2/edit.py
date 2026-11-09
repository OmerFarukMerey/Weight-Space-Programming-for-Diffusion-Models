"""Candidate c2: uniform eps gain 0.9875 on conv_out, plus a low-t-only time warp.

1. conv_out.weight *= G. Scales the predicted noise (guided eps) at every t.
2. time_embedding.linear_1.weight is re-fitted so that for integer t in [0, 999]
   W' s(t) ~= W s(f(t)), where s is the fixed sinusoidal embedding and
   f(t) = t - D * rise(t/100) * (1 - rise((t-300)/250)).
   So 100<t<300 is read as about t-40, and t>=550 is unchanged. Weighted least
   squares, keeping t>=550 essentially exact.
"""
import math
import torch

G = 0.9875
D = 40.0
LAM = 1.0


def rise(x):
    x = x.clamp(0, 1)
    return x * x * (3 - 2 * x)


def basis(t):
    half = 160
    om = torch.exp(-math.log(10000.0) * torch.arange(half, dtype=torch.float64) / half)
    a = t[:, None] * om[None, :]
    return torch.cat([torch.cos(a), torch.sin(a)], dim=1)  # flip_sin_to_cos layout


def warp(W):
    W = W.double()
    t = torch.arange(0, 1000, dtype=torch.float64)
    f = t - D * rise(t / 100) * (1 - rise((t - 300) / 250))
    q = torch.where(t < 100, 0.02, 1.0) * torch.where(t >= 550, 16.0, 1.0) * torch.where(t >= 995, 8.0, 1.0)
    S, Y = basis(t), basis(f) @ W.T
    A = S.T @ (q[:, None] * S) + LAM * torch.eye(320, dtype=torch.float64)
    return torch.linalg.solve(A, S.T @ (q[:, None] * Y) + LAM * W.T).T.float()


def apply(w):
    w["time_embedding.linear_1.weight"] = warp(w["time_embedding.linear_1.weight"])
    w["conv_out.weight"] *= G
