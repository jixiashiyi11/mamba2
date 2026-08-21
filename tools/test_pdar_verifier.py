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
    def __init__(self, full_context, prior_logits):
        super().__init__()
        self.full_context = full_context
        self.prior_logits = prior_logits

    def forward(self, tokens, spatial_shape):
        diluted_context = F.normalize(0.9 * tokens + 0.1 * self.full_context, dim=-1)
        return diluted_context, {
            "mamba_full_context_tokens": self.full_context,
            "mamba_context_logits": self.prior_logits,
            "mamba_global_prior": self.prior_logits,
        }


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
        calibration = F.interpolate(
            calibration,
            size=image_shape,
            mode="bilinear",
            align_corners=False,
        )
        mod_mask = feature_map.new_ones(feature_map.shape[0], 1, *feature_map.shape[-2:])
        return calibration.squeeze(1), mod_mask


def build_model(full_context, prior_logits):
    model = CLIPNormalityAD.__new__(CLIPNormalityAD)
    nn.Module.__init__(model)
    model.arcc = DummyARCC()
    model.mamba_context = DummyMamba(full_context, prior_logits)
    model.arcc_mode = "mamba_veto"
    model.arcc_inject_mamba = True
    model.arcc_mamba_fusion_mode = "add"
    model.arcc_mamba_feature_source = "pdar_delta"
    model.mamba_veto_source = "prior"
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
    model.mamba_context_mask_pool = "adaptive_max"
    model.mamba_context_outside_topk_ratio = 0.01
    model.mamba_context_separation_margin = 0.2
    model.margin = 0.2
    model._grid_size = lambda: (2, 2)
    model.training = False
    return model


def main():
    raw_vector = torch.tensor([1.0, 2.0, 3.0, 4.0])
    full_vector = torch.tensor([4.0, 1.0, 3.0, 2.0])
    tokens = raw_vector.view(1, 1, 4).repeat(2, 4, 1)
    full_context = full_vector.view(1, 1, 4).repeat(2, 4, 1)
    prior_logits = torch.tensor(
        [[[-2.0, 2.0], [-2.0, -2.0]], [[2.0, -2.0], [-2.0, -2.0]]]
    )
    model = build_model(full_context, prior_logits)
    raw_patch_map = torch.ones(2, 2, 2)
    raw_anomaly_map = torch.ones(2, 8, 8)
    protos = torch.stack((torch.zeros(2, 4), torch.ones(2, 4)), dim=1)

    _, debug, context_tokens = model._apply_arcc(
        tokens,
        raw_patch_map,
        raw_anomaly_map,
        image_shape=(8, 8),
        protos=protos,
        mamba_source_tokens=tokens,
    )
    expected_delta = F.layer_norm(full_context, (4,)) - F.layer_norm(tokens, (4,))
    expected_joint = tokens + 0.1 * expected_delta
    assert torch.allclose(context_tokens, expected_joint, atol=1e-6)
    assert torch.equal(debug["mamba_verifier_logits"], prior_logits)
    expected_support = torch.sigmoid(
        F.interpolate(
            prior_logits.unsqueeze(1),
            size=(8, 8),
            mode="bilinear",
            align_corners=False,
        ).squeeze(1)
    )
    assert torch.allclose(debug["mamba_support_map"], expected_support, atol=1e-6)

    masks = torch.zeros(1, 1, 4, 4)
    masks[:, :, :2, :2] = 1.0
    good_logits = torch.tensor([[[2.0, -2.0], [-2.0, -2.0]]])
    bad_logits = -good_logits
    good_loss = model._mamba_context_separation_loss(good_logits, masks)
    bad_loss = model._mamba_context_separation_loss(bad_logits, masks)
    assert torch.allclose(good_loss, good_loss.new_zeros(()))
    assert bad_loss > 4.0
    print("Full-PDAR injection and prior-verifier tests passed.")


if __name__ == "__main__":
    main()
