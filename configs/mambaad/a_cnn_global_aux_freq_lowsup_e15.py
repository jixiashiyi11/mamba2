from configs.mambaad.a_cnn_global_aux_frequency_dual_role_e15 import cfg as dual_role_cfg


class cfg(dual_role_cfg):
    def __init__(self):
        super(cfg, self).__init__()
        self.synthetic_local_anomaly.frequency_dual_role_force_mode = 'low_bg_nuisance'
        self.synthetic_local_anomaly.bce_weight = 0.0
        self.synthetic_local_anomaly.dice_weight = 0.0
        self.synthetic_local_anomaly.area_sparsity_weight = 0.0
