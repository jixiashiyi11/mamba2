from configs.mambaad.a_arcc_e2_feature_calib_e5 import cfg as e2_cfg


class cfg(e2_cfg):
    def __init__(self):
        super(cfg, self).__init__()

        # E2 backbone and losses, but disable ARCC calibration.
        self.model.kwargs.setdefault('arcc_kwargs', {})
        self.model.kwargs['arcc_kwargs'].update(
            use_arcc=False,
        )
