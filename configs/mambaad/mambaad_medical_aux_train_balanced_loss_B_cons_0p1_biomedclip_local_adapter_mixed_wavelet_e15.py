from configs.mambaad.mambaad_medical_aux_train_balanced_loss_B_cons_0p1_biomedclip_local_adapter_wavelet_e15 import cfg as wavelet_cfg


class cfg(wavelet_cfg):
    def __init__(self):
        super(cfg, self).__init__()

        self.synthetic_local_anomaly.lesion_mode = 'mixed_wavelet'
        self.synthetic_local_anomaly.wavelet_mix_prob = 0.5
