from configs.mambaad.a_arcc_e2_feature_calib_e5 import cfg as full_cfg


class cfg(full_cfg):
    def __init__(self):
        super(cfg, self).__init__()

        # Alias for the full ARCC-E2 loss setting used as the main method.
