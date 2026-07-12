from configs.mambaad.a_cnn_global_aux_e5 import cfg as base_cfg


class cfg(base_cfg):
    def __init__(self):
        super(cfg, self).__init__()

        self.model.kwargs.update(
            arcc_kwargs=dict(
                use_arcc=False,
            ),
        )
