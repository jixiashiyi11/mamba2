from configs.mambaad.mambaad_medical_aux_train_balanced_loss_B_cons_0p1_synthetic_local_anomaly import cfg as strong_cfg


class cfg(strong_cfg):
    def __init__(self):
        super(cfg, self).__init__()

        # Weaker synthetic-local-anomaly setting:
        # keep the same eval/model/training pipeline, but make the auxiliary
        # local anomaly signal smaller and softer so it does not dominate
        # brain/liver foreground structure.
        self.synthetic_local_anomaly.prob = 0.75
        self.synthetic_local_anomaly.loss_weight = 0.03
        self.synthetic_local_anomaly.min_area = 0.001
        self.synthetic_local_anomaly.max_area = 0.025
        self.synthetic_local_anomaly.noise_std = 0.08
        self.synthetic_local_anomaly.intensity_delta = 0.15

        self.model.kwargs['adaptive_mc_kwargs']['lambda_score_separation'] = 0.01
