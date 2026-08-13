from configs.mambaad.a_arcc_e2_loss_ab0_base_e5 import cfg as base_cfg


class cfg(base_cfg):
    def __init__(self):
        super(cfg, self).__init__()

        # Add synthetic-mask compactness constraints.
        self.synthetic_local_anomaly.outside_suppression_weight = 0.2
        self.synthetic_local_anomaly.area_sparsity_weight = 0.05
        self.synthetic_local_anomaly.outside_mask_dilate_iters = 2
        self.synthetic_local_anomaly.compact_mask_threshold = 0.05
        self.synthetic_local_anomaly.area_target_multiplier = 1.5
        self.synthetic_local_anomaly.area_target_slack = 0.005
