from configs.mambaad.a_arcc_e2_feature_calib_e5 import cfg as base_cfg


class cfg(base_cfg):
    def __init__(self):
        super(cfg, self).__init__()

        self.synthetic_local_anomaly.synthetic_generator_mode = 'synomaly_mask'
        self.synthetic_local_anomaly.use_morphology_prior = False
        self.synthetic_local_anomaly.use_frequency_appearance = False
        self.synthetic_local_anomaly.frequency_vis_dir = None
