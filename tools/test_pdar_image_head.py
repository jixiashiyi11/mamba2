import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def main():
    torch.manual_seed(7)
    batch, patches, channels = 3, 6, 4
    tokens = torch.randn(batch, patches, channels)
    verifier_logits = torch.randn(batch, patches)
    attention = torch.softmax((verifier_logits / 2.0).detach(), dim=1)

    pooled_mean = tokens.mean(dim=1)
    pooled_suspicious = torch.sum(attention.unsqueeze(-1) * tokens, dim=1)
    pooled = torch.cat((pooled_mean, pooled_suspicious), dim=-1)

    head = nn.Sequential(
        nn.LayerNorm(2 * channels),
        nn.Linear(2 * channels, channels),
        nn.SiLU(),
        nn.Dropout(0.1),
        nn.Linear(channels, 1),
    )
    nn.init.zeros_(head[-1].weight)
    nn.init.zeros_(head[-1].bias)

    scale = torch.sigmoid(torch.tensor(math.log(0.1 / 0.9)))
    legacy_score = torch.randn(batch)
    pdar_score = head(pooled).squeeze(1)
    combined_score = legacy_score + scale * pdar_score

    assert torch.allclose(attention.sum(dim=1), torch.ones(batch), atol=1e-6)
    assert torch.allclose(pdar_score, torch.zeros_like(pdar_score), atol=1e-6)
    assert torch.allclose(combined_score, legacy_score, atol=1e-6)
    assert torch.allclose(scale, scale.new_tensor(0.1), atol=1e-6)

    labels = torch.tensor([0.0, 1.0, 1.0])
    loss = F.binary_cross_entropy_with_logits(combined_score, labels)
    loss = loss + F.binary_cross_entropy_with_logits(pdar_score, labels)
    loss.backward()
    assert head[-1].weight.grad is not None
    assert torch.isfinite(head[-1].weight.grad).all()
    assert head[-1].weight.grad.abs().sum() > 0
    print("PDAR image-head pooling, zero-start, and gradient test passed.")


if __name__ == "__main__":
    main()
