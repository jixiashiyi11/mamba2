from configs.mambaad.a_cnn_global_mamba_residual_gate_rank_e15 import cfg as rank_cfg


class cfg(rank_cfg):
    def __init__(self):
        super(cfg, self).__init__()

        # Lightweight organ-wise calibration for the synthetic ranking loss.
        # Keys are matched exactly or as substrings of dataset class names.
        self.synthetic_local_anomaly.organ_calibration = {
            'brain': dict(
                score_temperature=0.12,
                hard_negative_margin=0.20,
                hard_negative_topk_ratio=0.015,
                hard_negative_pos_topk_ratio=0.35,
            ),
            'liver': dict(
                score_temperature=0.10,
                hard_negative_margin=0.30,
                hard_negative_topk_ratio=0.015,
                hard_negative_pos_topk_ratio=0.30,
            ),
            'retina': dict(
                score_temperature=0.12,
                hard_negative_margin=0.18,
                hard_negative_topk_ratio=0.035,
                hard_negative_pos_topk_ratio=0.25,
                hard_negative_min_pixels=2,
            ),
            'retinal': dict(
                score_temperature=0.12,
                hard_negative_margin=0.18,
                hard_negative_topk_ratio=0.035,
                hard_negative_pos_topk_ratio=0.25,
                hard_negative_min_pixels=2,
            ),
        }

        self.debug_gate_vis_dir = 'outputs/debug_gate_vis/mamba_residual_gate_rank_orgcal'
