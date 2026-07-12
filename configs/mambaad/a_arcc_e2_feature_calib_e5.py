from configs.mambaad.a_arcc_e0_baseline_e5 import cfg as base_cfg


class cfg(base_cfg):
    def __init__(self):
        super(cfg, self).__init__()

        self.model.kwargs.update(
            arcc_kwargs=dict(
                use_arcc=True,
                use_response=False,
                use_foreground=False,
                use_edge=False,
                lambda_init=0.1,
                kernel_size=3,
            ),
        )
