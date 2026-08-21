import math
from pathlib import Path
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model.clip_ad import CLIPNormalityAD


class DummyMamba(nn.Module):
    def forward(self, tokens, spatial_shape):
        return 2.0 * tokens, {}


class DummyARCC(nn.Module):
    def forward(
        self,
        feature_map,
        local_logits,
        foreground=None,
        edge=None,
        image_shape=None,
    ):
        calibration = -feature_map.square().mean(dim=1, keepdim=True)
        if image_shape is not None:
            calibration = F.interpolate(
                calibration,
                size=image_shape,
                mode="bilinear",
                align_corners=False,
            )
        mod_mask = feature_map.new_ones(feature_map.shape[0], 1, *feature_map.shape[-2:])
        return calibration.squeeze(1), mod_mask


def main():
    model = CLIPNormalityAD.__new__(CLIPNormalityAD)
    nn.Module.__init__(model)
    model.arcc = DummyARCC()
    model.mamba_context = DummyMamba()
    model.arcc_mode = "mamba_veto"
    model.arcc_inject_mamba = True
    model.arcc_mamba_fusion_mode = "add"
    model.arcc_mamba_feature_source = "context_tokens"
    model.mamba_veto_source = "semantic"
    model.arcc_mamba_projection = nn.Conv2d(4, 4, kernel_size=1, bias=False)
    model.arcc_mamba_fusion = None
    nn.init.eye_(model.arcc_mamba_projection.weight[:, :, 0, 0])
    model.arcc_mamba_injection_logit = nn.Parameter(
        torch.tensor(math.log(0.1 / 0.9))
    )
    model.arcc_lambda = nn.Parameter(torch.tensor(0.5))
    model.arcc_lambda_override = None
    model.mamba_veto_threshold = 0.0
    model.mamba_veto_temperature = 1.0
    model.mamba_veto_detach = True
    model.mamba_veto_alpha_logit = nn.Parameter(torch.tensor(math.log(0.1 / 0.9)))
    model._grid_size = lambda: (2, 2)
    model.training = False

    tokens = torch.ones(2, 4, 4)
    raw_patch_map = torch.ones(2, 2, 2)
    raw_anomaly_map = torch.ones(2, 8, 8)
    protos = torch.stack(
        (
            torch.zeros(2, 4),
            torch.ones(2, 4),
        ),
        dim=1,
    )
    final_map, debug, context_tokens = model._apply_arcc(
        tokens,
        raw_patch_map,
        raw_anomaly_map,
        image_shape=(8, 8),
        protos=protos,
        mamba_source_tokens=tokens,
    )

    gamma = debug["arcc_mamba_injection_gamma"]
    cnn_only = debug["arcc_cnn_only_final_map"]
    assert torch.allclose(gamma, gamma.new_tensor(0.1), atol=1e-6)
    assert torch.allclose(context_tokens, tokens + 0.1 * (2.0 * tokens), atol=1e-6)
    assert final_map.shape == raw_anomaly_map.shape
    assert cnn_only.shape == raw_anomaly_map.shape
    assert not torch.allclose(final_map, cnn_only)
    assert torch.all(final_map <= raw_anomaly_map + 1e-6)

    # The V7 concat projection starts from the same conservative numerical
    # fusion as addition, but can subsequently learn cross-source mixing.
    model.arcc_mamba_fusion_mode = "concat"
    model.arcc_mamba_projection = None
    model.arcc_mamba_fusion = nn.Conv2d(8, 4, kernel_size=1, bias=False)
    with torch.no_grad():
        model.arcc_mamba_fusion.weight.zero_()
        weight = model.arcc_mamba_fusion.weight[:, :, 0, 0]
        nn.init.eye_(weight[:, :4])
        nn.init.eye_(weight[:, 4:])
    _, concat_debug, concat_tokens = model._apply_arcc(
        tokens,
        raw_patch_map,
        raw_anomaly_map,
        image_shape=(8, 8),
        protos=protos,
        mamba_source_tokens=tokens,
    )
    assert torch.allclose(concat_tokens, tokens + 0.1 * (2.0 * tokens), atol=1e-6)
    assert "arcc_cnn_only_final_map" in concat_debug
    print("Controlled joint-ARCC counterfactual test passed.")


if __name__ == "__main__":
    main()
