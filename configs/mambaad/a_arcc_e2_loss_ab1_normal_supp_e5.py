from configs.mambaad.a_arcc_e2_loss_ab0_base_e5 import cfg as base_cfg


class cfg(base_cfg):
    def __init__(self):
        super(cfg, self).__init__()

        # Add normal-image suppression losses.
        self.model.kwargs['local_loss_kwargs'].update(
            normal_topk_loss_weight=0.4,
            background_loss_weight=0.05,
            edge_loss_weight=0.05,
            normal_topk_ratio=0.01,
        )
