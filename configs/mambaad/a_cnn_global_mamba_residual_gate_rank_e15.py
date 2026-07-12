from configs.mambaad.a_cnn_global_mamba_residual_gate_e15 import cfg as residual_gate_cfg


class cfg(residual_gate_cfg):
    def __init__(self):
        super(cfg, self).__init__()

        # AP/F1-oriented local ranking: lesion pixels should outrank hard
        # foreground negatives from the same anatomy.
        self.synthetic_local_anomaly.enabled = True
        self.synthetic_local_anomaly.hard_negative_ranking_weight = 0.5
        self.synthetic_local_anomaly.hard_negative_margin = 0.25
        self.synthetic_local_anomaly.hard_negative_topk_ratio = 0.02
        self.synthetic_local_anomaly.hard_negative_pos_topk_ratio = 0.30
        self.synthetic_local_anomaly.hard_negative_min_pixels = 4
        self.synthetic_local_anomaly.hard_negative_mask_threshold = 0.05
        self.synthetic_local_anomaly.hard_negative_dilate_iters = 2

        self.debug_gate_vis_dir = 'outputs/debug_gate_vis/mamba_residual_gate_rank'
