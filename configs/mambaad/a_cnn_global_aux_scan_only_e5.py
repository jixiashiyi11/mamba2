from configs.mambaad.a_cnn_global_aux_e5 import cfg as base_cfg


class cfg(base_cfg):
    def __init__(self):
        super(cfg, self).__init__()

        model_s = dict(self.model.kwargs.get('model_s', self.model_s))
        model_s.update(
            use_selective_scan=True,
            use_deformable_pool=False,
        )
        self.model_s = model_s
        self.model.kwargs['model_s'] = model_s

