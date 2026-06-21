from configs.mambaad.mambaad_medical_aux_train_balanced_loss_B_cons_0p1_biomedclip_local_adapter_mixed_wavelet_e15 import cfg as mixed_wavelet_cfg


class cfg(mixed_wavelet_cfg):
    def __init__(self):
        super(cfg, self).__init__()

        self.synthetic_local_anomaly.lesion_mode = 'mixed_wavelet'
        self.synthetic_local_anomaly.wavelet_mix_prob = 0.25

        self.synthetic_local_anomaly.min_area = 0.001
        self.synthetic_local_anomaly.max_area = 0.035
        self.synthetic_local_anomaly.noise_std = 0.08
        self.synthetic_local_anomaly.intensity_delta = 0.15

        self.synthetic_local_anomaly.wavelet_ll_delta = 0.08
        self.synthetic_local_anomaly.wavelet_edge_noise = 0.04
        self.synthetic_local_anomaly.wavelet_texture_noise = 0.03
        self.synthetic_local_anomaly.wavelet_texture_attenuation = 0.08

        self.model.kwargs['local_loss_kwargs'].update(
            normal_topk_loss_weight=0.2,
        )
