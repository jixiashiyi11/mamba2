from configs.mambaad.a_arcc_e2_feature_calib_e5 import cfg as base_cfg


class cfg(base_cfg):
    def __init__(self):
        super(cfg, self).__init__()

        # Frequency-role counterfactual setting:
        # positive roles should be localized as lesions, while nuisance roles
        # should be suppressed as false positives. Keep half of the normal
        # images unchanged so the model repeatedly sees clean normal foreground.
        self.synthetic_local_anomaly.enabled = True
        self.synthetic_local_anomaly.lesion_mode = 'frequency_dual_role'
        self.synthetic_local_anomaly.prob = 0.5

        self.synthetic_local_anomaly.loss_weight = 0.1
        self.synthetic_local_anomaly.bce_weight = 1.0
        self.synthetic_local_anomaly.dice_weight = 1.0
        self.synthetic_local_anomaly.suppress_loss_weight = 0.35
        self.synthetic_local_anomaly.frequency_role_ranking_weight = 0.25
        self.synthetic_local_anomaly.frequency_role_ranking_margin = 0.15
        self.synthetic_local_anomaly.frequency_role_ranking_topk_ratio = 0.30
        self.synthetic_local_anomaly.frequency_role_ranking_min_pixels = 4

        self.synthetic_local_anomaly.outside_suppression_weight = 0.15
        self.synthetic_local_anomaly.area_sparsity_weight = 0.03
        self.synthetic_local_anomaly.score_temperature = 0.1

        self.synthetic_local_anomaly.min_area = 0.001
        self.synthetic_local_anomaly.max_area = 0.035
        self.synthetic_local_anomaly.suppress_min_area = 0.002
        self.synthetic_local_anomaly.suppress_max_area = 0.060
        self.synthetic_local_anomaly.suppress_mask_threshold = 0.05
        self.synthetic_local_anomaly.nuisance_edge_dilate_iters = 1
        self.synthetic_local_anomaly.foreground_threshold = 5.0 / 255.0
        self.synthetic_local_anomaly.foreground_erode_iters = 1
        self.synthetic_local_anomaly.outside_mask_dilate_iters = 2
        self.synthetic_local_anomaly.compact_mask_threshold = 0.05
        self.synthetic_local_anomaly.area_target_multiplier = 1.5
        self.synthetic_local_anomaly.area_target_slack = 0.005

        # Frequency statistics measured on masked medical benchmarks:
        # brain:   low=0.6087, mid=0.2226, high=0.1687
        # liver:   low=0.3202, mid=0.2998, high=0.3800
        # retinal: low=0.3001, mid=0.3059, high=0.3940
        self.synthetic_local_anomaly.organ_frequency_prior = {
            'brain': 'low_dominant_blob',
            'liver': 'mixed_mid_high',
            'retinal': 'fine_mid_high',
        }
        self.synthetic_local_anomaly.frequency_dual_role_weights = {
            'brain': {
                'low_fg_lesion': 0.55,
                'low_bg_nuisance': 0.25,
                'high_fg_defect': 0.10,
                'high_edge_noise': 0.10,
            },
            'liver': {
                'low_fg_lesion': 0.25,
                'low_bg_nuisance': 0.15,
                'high_fg_defect': 0.35,
                'high_edge_noise': 0.25,
            },
            'retinal': {
                'low_fg_lesion': 0.05,
                'low_bg_nuisance': 0.20,
                'high_fg_defect': 0.45,
                'high_edge_noise': 0.30,
            },
            'default': {
                'low_fg_lesion': 0.25,
                'low_bg_nuisance': 0.20,
                'high_fg_defect': 0.35,
                'high_edge_noise': 0.20,
            },
        }

        self.synthetic_local_anomaly.frequency_low_delta = 0.055
        self.synthetic_local_anomaly.frequency_high_edge_noise = 0.025
        self.synthetic_local_anomaly.frequency_high_texture_noise = 0.025
        self.synthetic_local_anomaly.frequency_high_texture_gain = 0.10

        self.debug_eval = True
        self.debug_eval_dirname = 'debug_eval_freq_role'
        self.debug_vis_dirname = 'debug_vis_freq_role'
