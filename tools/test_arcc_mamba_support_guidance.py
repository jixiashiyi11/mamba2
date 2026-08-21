import math
from pathlib import Path
import sys
import types

import torch
import torch.nn as nn
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Load only CLIPNormalityAD and its direct modules. The repository's model
# package otherwise imports every benchmark model and tries to JIT-build an
# unrelated StyleGAN extension before this unit test can run.
model_package = types.ModuleType("model")
model_package.__path__ = [str(ROOT / "model")]


class _Registry:
    def register_module(self, cls):
        return cls


model_package.MODEL = _Registry()
sys.modules["model"] = model_package

from model.clip_ad import CLIPNormalityAD


class DummyMamba(nn.Module):
    def forward(self, tokens, spatial_shape):
        return tokens, {}


class DummyGuidedARCC(nn.Module):
    def __init__(self):
        super().__init__()
        self.last_feature_map = None
        self.last_support = None

    def forward(
        self,
        feature_map,
        local_logits,
        mamba_support=None,
        foreground=None,
        edge=None,
        image_shape=None,
    ):
        if mamba_support is None:
            raise AssertionError("Expected explicit Mamba support guidance.")
        self.last_feature_map = feature_map
        self.last_support = mamba_support
        calibration = -mamba_support.unsqueeze(1)
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
    model.arcc = DummyGuidedARCC()
    model.mamba_context = DummyMamba()
    model.arcc_mode = "mamba_veto"
    model.arcc_inject_mamba = False
    model.arcc_mamba_fusion_mode = "add"
    model.arcc_mamba_feature_source = "context_tokens"
    model.arcc_mamba_support_guidance = True
    model.arcc_mamba_support_detach = True
    model.mamba_veto_source = "semantic"
    model.arcc_lambda = nn.Parameter(torch.tensor(0.5))
    model.arcc_lambda_override = None
    model.mamba_veto_threshold = 0.0
    model.mamba_veto_temperature = 1.0
    model.mamba_veto_detach = True
    model.mamba_veto_alpha_logit = nn.Parameter(torch.tensor(math.log(0.1 / 0.9)))
    model.pdar_image_head = None
    model._grid_size = lambda: (2, 2)
    model.training = True

    tokens = torch.tensor(
        [[[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0],
          [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]],
        requires_grad=True,
    )
    raw_patch_map = torch.ones(1, 2, 2)
    raw_anomaly_map = torch.ones(1, 8, 8)
    protos = torch.stack(
        (
            torch.zeros(1, 4),
            torch.tensor([[1.0, -1.0, 2.0, -2.0]]),
        ),
        dim=1,
    )

    final_map, debug, _ = model._apply_arcc(
        tokens,
        raw_patch_map,
        raw_anomaly_map,
        image_shape=(8, 8),
        protos=protos,
        mamba_source_tokens=tokens,
    )

    expected_logits = torch.tensor([[[1.0, -1.0], [2.0, -2.0]]])
    expected_support = torch.sigmoid(expected_logits)
    assert torch.allclose(model.arcc.last_support, expected_support, atol=1e-6)
    assert not model.arcc.last_support.requires_grad
    assert torch.allclose(
        model.arcc.last_feature_map,
        tokens.transpose(1, 2).reshape(1, 4, 2, 2),
        atol=1e-6,
    )
    assert torch.allclose(debug["arcc_mamba_support_guidance"], expected_support)
    assert torch.all(final_map <= raw_anomaly_map + 1e-6)
    print("Detached Mamba-support ARCC guidance test passed.")


if __name__ == "__main__":
    main()
