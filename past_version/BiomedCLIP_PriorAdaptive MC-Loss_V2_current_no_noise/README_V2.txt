Saved from the current workspace state.

This V2 snapshot includes:
- Global semantic probe for `F_global`
- BiomedCLIP class-specific priors selected by `cls_name`
- Adaptive MC-Loss with:
  - compactness penalty
  - hard negative mining across abnormal priors
  - return_details support
- Trainer-side class EMA weighting / automatic class balancing
- Semantic probe dropout before `text_proj`

Important note:
- The Gaussian semantic noise has been removed in the current workspace.
- So this snapshot is a "current no-noise V2" version, not a reduced-norm noise version.

Saved files:
- configs/mambaad/mambaad_medical.py
- loss/adaptive_mc_loss.py
- model/mambaad.py
- trainer/mambaad_trainer.py
