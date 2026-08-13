import argparse
import math

import torch

from model.mambaad import CSSD, DepthAttentionResidual, PDARCSSD


def _assert_finite(name, tensor):
    if tensor is None:
        raise AssertionError(f"{name} did not receive a gradient.")
    if not torch.isfinite(tensor).all():
        raise AssertionError(f"{name} contains NaN or Inf.")


def _assert_nonzero_finite(name, tensor):
    _assert_finite(name, tensor)
    if tensor.detach().abs().sum().item() == 0:
        raise AssertionError(f"{name} is exactly zero.")


def _check_weight_sum(name, weights, depth_dim):
    expected = torch.ones_like(weights.sum(dim=depth_dim))
    torch.testing.assert_close(weights.sum(dim=depth_dim), expected, rtol=1e-5, atol=1e-6)
    print(f"{name} weight sum: OK")


def run_test(batch_size, grid_size, hidden_dim, device):
    torch.manual_seed(7)
    device = torch.device(device)
    shape = (batch_size, grid_size, grid_size, hidden_dim)
    token_count = grid_size * grid_size

    # Directly verify the official mixer contract: softmax is over depth N.
    mixer = DepthAttentionResidual(hidden_dim).to(device)
    sources = [torch.randn(shape, device=device) for _ in range(5)]
    mixed, raw_weights = mixer(sources)
    assert mixed.shape == shape
    assert raw_weights.shape == (5, batch_size, grid_size, grid_size)
    _check_weight_sum("raw final mixer", raw_weights, depth_dim=0)

    # Use the real LSSModule deformable branch. Selective scan is disabled so
    # this wiring/gradient test does not depend on the CUDA selective-scan op.
    pdar = PDARCSSD(
        hidden_dim=hidden_dim,
        grid_size=grid_size,
        depths=(1, 1, 1, 1),
        d_state=16,
        drop_path_rate=0.0,
        attn_drop_rate=0.0,
        scan_type="scan",
        num_direction=8,
        use_selective_scan=False,
        use_deformable_pool=True,
    ).to(device)
    assert all(stage.add_outer_residual is False for stage in pdar.stages)
    assert all(stage.use_adaln is False for stage in pdar.stages)

    tokens = torch.randn(batch_size, token_count, hidden_dim, device=device, requires_grad=True)
    semantic = torch.randn(batch_size, hidden_dim, device=device)
    output, debug = pdar(tokens, semantic, (grid_size, grid_size), return_debug=True)

    assert output.shape == (batch_size, token_count, hidden_dim)
    assert debug["depth_final_context"].shape == (batch_size, token_count, hidden_dim)
    source_counts = [weights.shape[1] for weights in debug["depth_stage_weights"]]
    final_source_count = debug["depth_final_weights"].shape[1]
    assert source_counts == [1, 2, 3, 4], source_counts
    assert final_source_count == 5, final_source_count
    print(f"stage history source counts: {source_counts}")
    print(f"final history source count: {final_source_count}")

    for idx, weights in enumerate(debug["depth_stage_weights"], start=1):
        _check_weight_sum(f"stage {idx}", weights, depth_dim=1)
    _check_weight_sum("final", debug["depth_final_weights"], depth_dim=1)

    target = torch.randn_like(output)
    loss = (output.float() - target.float()).square().mean()
    if not math.isfinite(loss.item()):
        raise AssertionError(f"loss is not finite: {loss.item()}")
    loss.backward()
    print(f"loss: {loss.item():.8f}")

    for idx, depth_mixer in enumerate(pdar.depth_mixers, start=2):
        _assert_nonzero_finite(f"stage {idx} proj.weight.grad", depth_mixer.proj.weight.grad)
    _assert_nonzero_finite("final proj.weight.grad", pdar.final_depth_mixer.proj.weight.grad)

    lss_grads = [
        parameter.grad
        for stage in pdar.stages
        for parameter in stage.parameters()
        if parameter.requires_grad and parameter.grad is not None
    ]
    if not lss_grads:
        raise AssertionError("No LSS parameter received a gradient.")
    for idx, grad in enumerate(lss_grads):
        _assert_finite(f"LSS grad {idx}", grad)
    if sum(grad.detach().abs().sum().item() for grad in lss_grads) == 0:
        raise AssertionError("All LSS gradients are exactly zero.")
    print(f"query gradients: OK ({len(pdar.depth_mixers) + 1} mixers)")
    print(f"LSS gradients: OK ({len(lss_grads)} tensors)")

    # Baseline CSSD must retain its original outer residual behavior and output path.
    baseline = CSSD(
        hidden_dim=hidden_dim,
        grid_size=grid_size,
        depths=(1, 1, 1, 1),
        d_state=16,
        drop_path_rate=0.0,
        attn_drop_rate=0.0,
        scan_type="scan",
        num_direction=8,
        use_selective_scan=False,
        use_deformable_pool=True,
    ).to(device)
    assert all(stage.add_outer_residual is True for stage in baseline.stages)
    assert all(stage.use_adaln is True for stage in baseline.stages)
    with torch.no_grad():
        baseline_output = baseline(tokens.detach(), semantic, (grid_size, grid_size))
    assert baseline_output.shape == (batch_size, token_count, hidden_dim)
    print("baseline CSSD forward: OK")
    print(f"PDAR output shape: {tuple(output.shape)}")
    print(f"final context shape: {tuple(debug['depth_final_context'].shape)}")


def main():
    parser = argparse.ArgumentParser(description="Verify official AttnRes semantics in PDAR-CSSD.")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grid-size", type=int, default=24)
    parser.add_argument("--hidden-dim", type=int, default=768)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    run_test(args.batch_size, args.grid_size, args.hidden_dim, args.device)


if __name__ == "__main__":
    main()
