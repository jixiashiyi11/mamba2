from configs.mambaad.a_arcc_e3_response_calib_e5 import cfg as base_cfg


class cfg(base_cfg):
    def __init__(self):
        super(cfg, self).__init__()

        # Roughly half of the training images remain unchanged with all-zero
        # synthetic masks, so normal foreground is explicitly trained as normal.
        self.synthetic_local_anomaly.prob = 0.5
