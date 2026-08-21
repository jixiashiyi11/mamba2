#!/usr/bin/env python3
"""Focused invariants for the probability-domain Mamba veto."""

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from util.mamba_veto import apply_mamba_probability_veto


def main():
    raw = torch.tensor([[-2.0, -0.2, 0.4, 2.0]], requires_grad=True)
    bidirectional = torch.tensor([[-1.0, -0.5, 1.2, 1.0]], requires_grad=True)
    support = torch.tensor([[0.0, 0.25, 0.75, 1.0]])
    alpha = torch.tensor(0.4, requires_grad=True)

    final, suppressed, veto = apply_mamba_probability_veto(
        raw, bidirectional, support, alpha
    )

    assert torch.all(suppressed <= raw), "ARCC suppression must not exceed A_raw."
    assert torch.all(final <= suppressed + 1e-6), "Mamba veto must never amplify."
    assert torch.all(final <= raw + 1e-6), "A_final must never exceed A_raw."
    assert torch.allclose(final[:, -1], suppressed[:, -1], atol=1e-6), (
        "Full Mamba support must preserve the suppressed candidate."
    )
    assert torch.all(veto[:, :-1] > 0), "Weak support must produce an active veto."

    final.sum().backward()
    assert raw.grad is not None
    assert bidirectional.grad is not None
    assert alpha.grad is not None
    print("Mamba veto invariants passed.")


if __name__ == "__main__":
    main()
