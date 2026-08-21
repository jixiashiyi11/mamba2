#!/usr/bin/env python3
"""Verify that patch-level Mamba targets preserve tiny positive masks."""

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from util.mamba_veto import resize_mamba_patch_targets


def main():
    mask = torch.zeros(1, 1, 28, 28)
    mask[0, 0, 13, 13] = 1.0

    nearest = resize_mamba_patch_targets(mask, (2, 2), mode="nearest")
    adaptive_max = resize_mamba_patch_targets(mask, (2, 2), mode="adaptive_max")

    assert nearest.sum().item() == 0.0, "The fixture must expose nearest-neighbor loss."
    assert adaptive_max.sum().item() == 1.0, "Adaptive max pooling must preserve the defect."
    assert adaptive_max[0, 0, 0, 0].item() == 1.0
    print("Mamba adaptive-max mask target test passed.")


if __name__ == "__main__":
    main()
