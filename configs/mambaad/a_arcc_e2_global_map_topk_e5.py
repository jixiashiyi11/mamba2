from configs.mambaad.a_arcc_e2_feature_calib_e5 import cfg as base_cfg


class cfg(base_cfg):
    def __init__(self):
        super(cfg, self).__init__()

        self.model.kwargs.setdefault('image_branch_kwargs', {})
        self.model.kwargs['image_branch_kwargs'].update(
            image_score_source='global_map_topk',
            image_score_beta=0.2,
            map_topk_ratio=0.01,
        )
