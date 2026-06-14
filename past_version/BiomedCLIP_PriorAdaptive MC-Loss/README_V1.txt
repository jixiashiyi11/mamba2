Reconstructed V1-B from dialogue history.

Scope of this saved version:
- Global semantic probe (GAP + text_proj + L2 normalize)
- BiomedCLIP priors loaded once in trainer
- Per-class prior selection by cls_name
- Basic Adaptive MC-Loss with adaptive margin
- Original teacher-student multi-layer reconstruction loss preserved

Not included in this V1 snapshot:
- compactness penalty (beta / loss_compact)
- hard negative mining (s_neg_all / s_neg_max)
- class EMA weighting / ACW
- text dropout in semantic probe
- Gaussian noise perturbation

Saved files:
- configs/mambaad/mambaad_medical.py
- loss/adaptive_mc_loss.py
- model/mambaad.py
- trainer/mambaad_trainer.py
