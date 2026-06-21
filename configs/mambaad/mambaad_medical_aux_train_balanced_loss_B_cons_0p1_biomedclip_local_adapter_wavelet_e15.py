from configs.mambaad.mambaad_medical_aux_train_balanced_loss_B_cons_0p1_biomedclip_local_adapter_e25 import cfg as base_cfg


class cfg(base_cfg):
    def __init__(self):
        super(cfg, self).__init__()

        self.epoch_full = 15
        self.test_start_epoch = 5
        self.test_per_epoch = 5
        self.trainer.epoch_full = self.epoch_full
        self.trainer.test_start_epoch = self.test_start_epoch
        self.trainer.test_per_epoch = self.test_per_epoch
        self.trainer.scheduler_kwargs['decay_epochs'] = int(self.epoch_full * 0.8)

        self.synthetic_local_anomaly.lesion_mode = 'wavelet'
        self.synthetic_local_anomaly.prob = 1.0
        self.synthetic_local_anomaly.loss_weight = 1.0
        self.synthetic_local_anomaly.min_area = 0.005
        self.synthetic_local_anomaly.max_area = 0.08
        self.synthetic_local_anomaly.noise_std = 0.12
        self.synthetic_local_anomaly.intensity_delta = 0.25
        self.synthetic_local_anomaly.wavelet_ll_delta = 0.12
        self.synthetic_local_anomaly.wavelet_edge_noise = 0.06
        self.synthetic_local_anomaly.wavelet_texture_noise = 0.05
        self.synthetic_local_anomaly.wavelet_texture_attenuation = 0.15
