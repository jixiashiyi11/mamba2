from configs.mambaad.a_cnn_global_aux_e15 import cfg as cnn_global_aux_cfg


class cfg(cnn_global_aux_cfg):
    def __init__(self):
        super(cfg, self).__init__()

        self.synthetic_local_anomaly.lesion_mode = 'multi_weak'

        # About 30% of normal-only samples are kept unchanged. This prevents the
        # adapter from treating one synthetic style as the definition of anomaly.
        self.synthetic_local_anomaly.prob = 0.7

        # Small, soft, foreground-limited pseudo lesions.
        self.synthetic_local_anomaly.min_area = 0.001
        self.synthetic_local_anomaly.max_area = 0.035
        self.synthetic_local_anomaly.foreground_erode_iters = 1
        self.synthetic_local_anomaly.num_blobs_min = 1
        self.synthetic_local_anomaly.num_blobs_max = 3
        self.synthetic_local_anomaly.soft_edge_power = 1.6
        self.synthetic_local_anomaly.soft_mask_threshold = 0.03

        # Multi-type weak synthetic anomaly mixture:
        # spatial blob, mild wavelet, blur/sharpness, contrast, local copy-paste.
        self.synthetic_local_anomaly.multi_weak_weights = (0.20, 0.25, 0.20, 0.20, 0.15)

        self.synthetic_local_anomaly.noise_std = 0.04
        self.synthetic_local_anomaly.intensity_delta = 0.12

        self.synthetic_local_anomaly.wavelet_ll_delta = 0.06
        self.synthetic_local_anomaly.wavelet_edge_noise = 0.025
        self.synthetic_local_anomaly.wavelet_texture_noise = 0.02
        self.synthetic_local_anomaly.wavelet_texture_attenuation = 0.06

        self.synthetic_local_anomaly.blur_kernel_size = 5
        self.synthetic_local_anomaly.blur_strength = 0.30
        self.synthetic_local_anomaly.sharp_amount = 0.25

        self.synthetic_local_anomaly.contrast_delta = 0.16
        self.synthetic_local_anomaly.brightness_delta = 0.05

        self.synthetic_local_anomaly.copypaste_max_shift = 0.10
        self.synthetic_local_anomaly.copypaste_strength = 0.35

