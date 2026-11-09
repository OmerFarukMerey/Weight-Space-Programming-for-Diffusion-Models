import math, torch

# (t, delta): the time embedding sees t - delta(t), piecewise linear in t.
KNOTS = [(0,0),(124,60),(166,85),(199,100),(249,70),(400,70),(470,0),(1000,0)]

def _emb(t):
    half = 160
    freqs = torch.exp(-math.log(10000) * torch.arange(half, dtype=torch.float64) / half)
    args = t[:, None] * freqs[None]
    return torch.cat([torch.cos(args), torch.sin(args)], 1)

def _delta(t):
    xs = torch.tensor([k[0] for k in KNOTS], dtype=torch.float64)
    ys = torch.tensor([k[1] for k in KNOTS], dtype=torch.float64)
    t = t.clamp(xs[0], xs[-1])
    i = torch.searchsorted(xs, t, right=True).clamp(1, len(xs) - 1)
    u = (t - xs[i - 1]) / (xs[i] - xs[i - 1])
    return ys[i - 1] + u * (ys[i] - ys[i - 1])

def apply(w):
    # Closed-form ridge fit of the smallest change dL to time_embedding.linear_1 such that
    # (L + dL) S(t) = L S(t - delta(t)) on integer t, weighted toward the timesteps that
    # 5, 6 and 8 step trailing sampling actually visits.
    L = w["time_embedding.linear_1.weight"]
    Ld = L.double()
    ts = torch.arange(0, 1000, dtype=torch.float64)
    used = torch.tensor([999, 874, 832, 799, 749, 666, 624, 599, 499, 399, 374, 332, 249, 199, 166, 124])
    weight = torch.ones_like(ts)
    weight[used] = 200.0
    S = _emb(ts)
    resid = (_emb(ts - _delta(ts)) - S) @ Ld.T
    A = (S * weight[:, None]).T @ S + 10.0 * torch.eye(320, dtype=torch.float64)
    B = (S * weight[:, None]).T @ resid
    L.copy_((Ld + torch.linalg.solve(A, B).T).float())
