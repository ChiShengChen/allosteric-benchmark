#!/usr/bin/env python
"""A residue-graph message-passing network.

Deliberately small. The generalisation argument that has governed every model in
this repository applies here too: ~90 labelled proteins is not many, and a network
with more capacity will fit the training folds and lose the test ones. Hidden
width 24 over 4 layers is about 15k parameters; the sweep in `run.py` reports
whether more helps, rather than assuming it does.

Depth is a physical choice, not a hyperparameter. Each layer moves information one
hop on a 10 A contact graph, so 4 layers reach roughly 40 A -- the scale an
allosteric signal has to travel. Going deeper buys reach at the cost of
oversmoothing, which the sweep also tests.

The architecture is the standard one: an edge-conditioned message, a residual node
update, degree normalisation so that buried residues do not simply accumulate more
signal than exposed ones. That last point matters here -- burial is one of this
benchmark's trivial controls, and a network that ranked residues by how much
message they received would be reproducing it.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class MPNN(nn.Module):
    def __init__(self, in_dim=3, hidden=24, layers=4, edge_dim=16):
        super().__init__()
        self.embed = nn.Linear(in_dim, hidden)
        self.msg = nn.ModuleList([
            nn.Sequential(nn.Linear(hidden + edge_dim, hidden), nn.SiLU(),
                          nn.Linear(hidden, hidden))
            for _ in range(layers)])
        self.upd = nn.ModuleList([
            nn.Sequential(nn.Linear(2 * hidden, hidden), nn.SiLU(),
                          nn.Linear(hidden, hidden))
            for _ in range(layers)])
        self.read = nn.Sequential(nn.Linear(hidden, hidden), nn.SiLU(),
                                  nn.Linear(hidden, 1))

    def forward(self, x, src, dst, eattr):
        h = self.embed(x)
        n = x.shape[0]
        deg = torch.zeros(n, device=x.device).index_add_(
            0, dst, torch.ones_like(dst, dtype=x.dtype)).clamp(min=1.0)
        inv = deg.rsqrt().unsqueeze(-1)
        for msg, upd in zip(self.msg, self.upd):
            m = msg(torch.cat([h[src], eattr], dim=-1))
            agg = torch.zeros_like(h).index_add_(0, dst, m) * inv
            h = h + upd(torch.cat([h, agg], dim=-1))
        return self.read(h).squeeze(-1)

    @property
    def n_params(self):
        return sum(p.numel() for p in self.parameters())
