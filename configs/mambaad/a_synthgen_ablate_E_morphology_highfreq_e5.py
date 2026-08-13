from configs.mambaad.a_arcc_e2_feature_calib_e5 import cfg as base_cfg


class cfg(base_cfg):
    def __init__(self):
        super(cfg, self).__init__()

        self.synthetic_local_anomaly.synthetic_generator_mode = 'morphology_frequency'
        self.synthetic_local_anomaly.use_morphology_prior = True
        self.synthetic_local_anomaly.morphology_prior_path = 'assets/morphology_prior.json'
        self.synthetic_local_anomaly.use_frequency_appearance = True
        self.synthetic_local_anomaly.frequency_appearance_mode = 'high'
        self.synthetic_local_anomaly.frequency_vis_dir = 'outputs/frequency_synthesis_vis/ablate_E_high'
