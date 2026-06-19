from configs.mambaad.mambaad_medical_aux_train_balanced_loss_B_cons_0p1_synthetic_local_anomaly import cfg as base_cfg


class cfg(base_cfg):
    def __init__(self):
        super(cfg, self).__init__()

        # Pure-ZSAD brain-oriented synthetic anomaly:
        # no target brain images are added; this only changes the synthetic
        # auxiliary signal to be smaller, softer, and less edge-biased.
        self.synthetic_local_anomaly.lesion_mode = 'soft_brain'
        self.synthetic_local_anomaly.prob = 0.9
        self.synthetic_local_anomaly.loss_weight = 0.05
        self.synthetic_local_anomaly.min_area = 0.0008
        self.synthetic_local_anomaly.max_area = 0.018
        self.synthetic_local_anomaly.noise_std = 0.06
        self.synthetic_local_anomaly.intensity_delta = 0.18
        self.synthetic_local_anomaly.foreground_erode_iters = 4
        self.synthetic_local_anomaly.num_blobs_min = 1
        self.synthetic_local_anomaly.num_blobs_max = 3
        self.synthetic_local_anomaly.soft_edge_power = 1.8
        self.synthetic_local_anomaly.soft_mask_threshold = 0.04

        self.model.kwargs['adaptive_mc_kwargs']['lambda_score_separation'] = 0.012

        self.epoch_full = 25
        self.test_start_epoch = 25
        self.test_per_epoch = 5
        self.trainer.epoch_full = self.epoch_full
        self.trainer.test_start_epoch = self.test_start_epoch
        self.trainer.test_per_epoch = self.test_per_epoch
        self.trainer.scheduler_kwargs['decay_epochs'] = int(self.epoch_full * 0.8)
